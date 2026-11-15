# GUARDBOUND — SINGLE HANDOFF REPORT

One document. Everything needed to write the next prompt. Evidence lives in the
Phase 22-27 reports at the repo root; `docs/audit/` holds earlier phases.

**Status: `READY FOR MINIMAL QUALIFICATION`** (Phase 27 fail-closed gate).

The three Phase 23 P0 fixes remain verified, the qualification configuration matches
R1, and the qualification verdict is a real gate: the NBF readiness checks are a
mandatory `local` decision, per-probe statuses drive `probes_complete`, and the gate
now **fails closed** — any local verdict other than `QUALIFIED` (including
`INSUFFICIENT_DATA`, i.e. `--skip-nbf`) stops the run before a provider client is
constructed, with a persisted report, `stopped_before_paid_probes: true` and zero
usage calls. `--model` and `--attacker-prompt-condition` are rejected on paper
configs, the local NBF gate runs before any client, and per-call usage is persisted
incrementally.

**No paid request has ever been made, and this status does not authorise one.**

`main` @ `89ed03f` + uncommitted work (Phases 20-27).

Read first: `PHASE_27_FINAL_FAIL_CLOSED_GATE_REPORT.md` (gate semantics, evidence,
the exact paid command), then `PHASE_26_FINAL_QUALIFICATION_READINESS_REPORT.md` for
the probe model and `PHASE_24_P0_FIX_REPORT.md` for the D-01…D-16 mapping.

---

## 1. What Phase 24 changed

| ID | Was | Now |
|---|---|---|
| **D-01** | attack-level NBF filter used `h < +η` (η = 1e-3) while the paper uses `h < −η` | one rule module (`defense/filter_rule.py`); `threshold_from_eta` + `admits` used by every filter site; pilot converts once. Boundary table verified for `0, −5e-4, −1e-3, −1.5e-3, −2e-3, +1e-3` |
| **D-02** | composite ActorAttack had no `steer_target` and never installed the candidate filter | `steer_target` / `use_safety_filter` / `safety_threshold_override`; mirrors the generic runner; `ValueError` instead of a silent no-op; control flow unchanged |
| **D-03** | failed goals entered the ASR denominator as 0-turn conversations | `judge_asr.load_corpus_split` excludes infrastructure failures, invalid evaluator outcomes and turn-less rows; counts reported; zero valid goals → exit 2, no report |
| **D-04** | manifest recorded legacy prompt hashes | hashes resolve the configured variant; rubric hash added; legacy block retained |
| **D-05** | qualification tested mini/0.0/legacy/η=0 | `configs/qualification_r1_gpt4o.yaml` (gpt-4o, 0.7×3, official, η 0.001) + temperatures/variant threaded into probes + ActorAttack pre-attack probe |
| **D-06/07/08** | no dirty flag, dataset hash, or completion facts | manifest records paper, commit, `git_dirty`, status hash, dataset sha256 + goal-id hash, configured vs selected counts, timestamps, `complete`, per-category counts, ASR denominator, usage paths |
| **D-09** | `max(scores)` raised on `None` | `aggregate_scores`; invalid outcomes counted, never safe/unsuccessful |

---

## 2. Verified and must not be re-litigated

| Item | Evidence |
|---|---|
| NBF rule at the boundary (both mechanisms) | `PHASE_24_NBF_THRESHOLD_VALIDATION.md` |
| ActorAttack steering on/off, filter reached | `PHASE_24_ACTORATTACK_STEERING_AUDIT.md` |
| ASR denominator (failures excluded, 0 % impossible) | `PHASE_24_ASR_AGGREGATION_AUDIT.md` |
| Qualification stack == R1 | `PHASE_24_QUALIFICATION_ALIGNMENT.md` |
| Manifest completeness | `PHASE_24_MANIFEST_REPRODUCIBILITY.md` |
| Dataset 200 behaviours, sha256 `ac789de8…` | Phase 23 dataset report |
| Checkpoint `cea1a75b…a136fe`, 1536-32-32-5 / 1536-512-512-768 | Phase 23 NBF report |
| Prompt bytes on the paper path (`paper_path_faithful = true`) | `scripts/audit_prompt_fidelity.py` |
| Frozen paths clean | `git status --porcelain` empty for `results/ runs/ checkpoints/ data/ reproduction*.yaml nbf_original_stuff` |

Tests: **1131 collected / 1127 passed / 4 failed / 0 errors / 0 skipped**. The four
are pre-existing and environmental (2 CUDA-gated, 2 missing `Qwen/Qwen3.5-4B`
weights). Do not "fix" them; do not read a green suite as methodology evidence.

---

## 3. Next commands (all free)

```powershell
Set-Location "C:\Users\CSLAB\Desktop\hemanth\Guardbound"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
$env:HF_HUB_OFFLINE = "1"     # optional: keeps the embedder load off the network

python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=line --junitxml=_snapF.xml
python -m pytest tests/test_phase24_p0_fixes.py -q
python scripts/audit_prompt_fidelity.py
python scripts/cloud_pilot.py --config configs/paper_reproduction_r1_gpt4o.yaml `
    --variant r1_paper --dry-run --limit 2
python scripts/cloud_model_qualification.py --config configs/qualification_r1_gpt4o.yaml --dry-run
```

Paid (needs explicit approval, and the key exported in the shell only):

```powershell
python scripts/cloud_model_qualification.py --config configs/qualification_r1_gpt4o.yaml `
    --confirm-paid-request
```

There is no `--budget-usd`; the cap was removed in Phase 22 and must not return.

---

## 4. Hard constraints (unchanged)

- No paid request, smoke test, qualification or experiment without explicit approval.
- Never print, log or pass the API key as an argument.
- Do not edit attack algorithms, evaluator rubrics, NBF architecture/checkpoint/scoring.
- Do not touch `results/`, Phase 18 / 18-ext / 19 scripts, or `configs/reproduction*.yaml`.
- No model substitution. C1/I1 stays deferred. No R2 sweep. No budget cap.

---

## 5. Key facts

| Item | Value |
|---|---|
| R1 config | `configs/paper_reproduction_r1_gpt4o.yaml` |
| Qualification config | `configs/qualification_r1_gpt4o.yaml` |
| Models | `gpt-4o-2024-08-06` ×3 — Deprecated, shutdown 2027-01-14 |
| Temperatures | 0.7 attacker / 0.7 target / 0.7 evaluator |
| NBF rule | filter iff `h + η ≥ 0`; η = 0.001 ⇒ admit iff `h < −0.001` |
| Max turns / goals / unsafe label | 8 / 200 / 5 |
| Prompts | `official` variant, byte-identical to the release |
| Paper gaps | no seed, no repetitions, no CIs, no token limits, no API budget |
| Test baseline | 1131 / 1127 / 4 / 0 |

---

## 6. Where things are

| Path | What |
|---|---|
| `PHASE_24_FINAL_READINESS_REPORT.md` | **decision report** (READY FOR MINIMAL QUALIFICATION) |
| `PHASE_24_P0_FIX_REPORT.md` | every fix + the D-01…D-16 status map |
| `PHASE_24_{NBF_THRESHOLD_VALIDATION,ACTORATTACK_STEERING_AUDIT,ASR_AGGREGATION_AUDIT,QUALIFICATION_ALIGNMENT,MANIFEST_REPRODUCIBILITY,TEST_RESULTS}.md` | per-defect evidence |
| `PHASE_23_*` | the audit that found the defects (evidence, pre-fix state) |
| `src/guardbound/defense/filter_rule.py` | the single NBF decision rule |
| `src/guardbound/attacks/runner.py` | generic loop + `run_actor_attack_composite` |
| `src/guardbound/paper_protocol.py` | R1 invariants (incl. threshold == eta) |
| `scripts/{cloud_pilot,cloud_model_qualification,judge_asr,_cloud_common}.py` | execution, qualification, ASR loader, manifest helpers |
| `tests/test_phase24_p0_fixes.py` | 64 regression tests for this phase |
