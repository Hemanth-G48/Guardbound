# PHASE 24 — P0 FIX REPORT

Defect correction for the three P0 findings of the Phase 23 audit, plus the P1/P2
gaps that make the R1 path reproducible. Offline only: no paid request, no
qualification, no pilot, no experiment.

**Result: `READY FOR MINIMAL QUALIFICATION`** (implementation). No paid run is
authorised by this document.

---

## 1. What changed, in one table

| ID | Defect | Fix | Files | Tests |
|---|---|---|---|---|
| **D-01** | attack-level NBF filter inverted η (`h < +0.001` instead of `h < −0.001`) | one shared rule module; `threshold_from_eta` at every η→threshold boundary; `admits(...)` at every comparison | `defense/filter_rule.py` (new), `defense/steered_chat.py`, `attacks/runner.py`, `attacks/actor_attack.py`, `attacks/opposite_day.py`, `attacks/acronym.py`, `paper_protocol.py`, `scripts/_cloud_common.py`, `scripts/cloud_pilot.py` | 17 |
| **D-02** | composite ActorAttack had no `steer_target`, never installed the attack-level filter | explicit `steer_target` / `use_safety_filter` / `safety_threshold_override`; mirrors the generic runner; loud `ValueError` instead of a silent no-op | `attacks/runner.py`, `scripts/cloud_pilot.py` | 9 |
| **D-03** | failed goals entered the ASR denominator as 0-turn conversations | `load_corpus_split` excludes infrastructure failures, invalid evaluator outcomes and turn-less rows; counted separately; zero denominator → exit 2, no report | `scripts/judge_asr.py`, `scripts/cloud_pilot.py`, `evaluation/asr.py` | 6 |
| **D-04** | manifest recorded legacy prompt hashes; no rubric hash | fingerprints resolve the configured variant; rubric hash; legacy block retained | `attacks/prompt_variants.py`, `llm/attacker_arms.py`, `scripts/_cloud_common.py` | 3 |
| **D-05** | qualification harness tested a non-R1 stack | `configs/qualification_r1_gpt4o.yaml`; per-role temperatures + prompt variant threaded into every probe; ActorAttack pre-attack probe; R1 invariants enforced | `scripts/cloud_model_qualification.py`, `configs/qualification_r1_gpt4o.yaml` (new), `configs/cloud_qualification.yaml` | 10 |
| **D-06** | no dirty-git state | `git_dirty()`, `git_status_sha256()` recorded in the manifest | `scripts/_cloud_common.py`, `scripts/cloud_pilot.py` | 1 |
| **D-07** | no dataset SHA-256 / goal hash; `num_goals_requested` was post-`--limit` | `dataset_provenance()` records sha256, goal-id hash, configured vs selected counts, sampling, shuffle; `limit` recorded | same | 2 |
| **D-08** | no timestamps, completion marker, counts or ASR denominator | manifest `run` block, rewritten after the loop; non-zero exit on incomplete/invalid runs | `scripts/cloud_pilot.py` | 1 |
| **D-09** | `max(scores)` raised on `None` and discarded a completed conversation | `aggregate_scores()`; invalid scores counted, never treated as safe/unsuccessful | `scripts/cloud_pilot.py` | 3 |
| **D-13** | inverted score docstrings | corrected (`higher = more unsafe`) | `attacks/runner.py`, `attacks/actor_attack.py` | n/a |

Totals: 10 files changed, 3 new files, 64 new tests, suite 1067 → 1131.

---

## 2. D-01 — NBF threshold (detail)

`PHASE_24_NBF_THRESHOLD_VALIDATION.md`.

**Root cause.** The configuration's `nbf.threshold` is the paper's non-negative
margin η, but the runners take a **raw admission threshold**. The pilot passed
`+0.001` where `−0.001` was required, and the composite path never negated at all.

**Before → after, executed with a uniform-logit barrier (`h = 0.0` exactly):**

| Path | Before (η = 1e-3) | After |
|---|---|---|
| generic runner + Crescendo, `steer_target=False` | 3 target calls (admitted) | **0 target calls** (filtered) |
| `ActorAttack.call_multi` / `summary` | `+η` passed, admitted | `−η`, filtered |
| `SteeredLLMChat` | filtered (already correct) | filtered (same rule, shared helper) |
| `round_number` guard | unchanged | unchanged |

**Boundary table** (`η = 0.001`, both mechanisms):

| h | 0.0 | −0.0005 | −0.001 | −0.0015 | −0.002 | +0.001 |
|---|---|---|---|---|---|---|
| decision | Filter | Filter | Filter | Admit | Admit | Filter |

**Invariants.** `threshold_from_eta(η) = −η`; `h = −η` is filtered (strict `<`);
a config whose `threshold ≠ eta` or whose `eta < 0` is rejected; η = 0 (every
frozen run) is bit-identical to before.

**Frozen artifacts.** `git status --porcelain -- results runs checkpoints data
configs/reproduction*.yaml nbf_original_stuff` → empty. Checkpoint hash
`cea1a75b…a136fe` unchanged; no architecture, embedding, state or score change.

---

## 3. D-02 — ActorAttack steering (detail)

`PHASE_24_ACTORATTACK_STEERING_AUDIT.md`.

**Root cause.** The composite runner's steered wrapper was keyed on "a barrier was
supplied" only, and `set_safety_filter` was never called — so the official
candidate filter was unreachable and `use_safety_filter=True` was a silent no-op.

**Before → after:**

| Aspect | Before | After |
|---|---|---|
| `steer_target` | not a parameter (`TypeError`) | accepted and honoured |
| R1 (`steer_target: false`) | target-steered | candidate filter installed, **no** steered wrapper |
| `set_safety_filter` | never called | called whenever a barrier is supplied |
| `use_safety_filter=True` without a barrier | silent no-op | `ValueError` |
| Official control flow | preserved | preserved (all prior contract tests pass) |

**Executed.** Steering-off: wrapper constructed 0×, `_safety_threshold = −0.001`,
candidate scores recorded, unsafe query never reaches the target. Steering-on:
wrapper constructed 1× with `eta=0.001`, `chat` called, and a query the candidate
filter admits but the Q-filter rejects never reaches the target.

**Frozen artifacts.** No change to `ActorAttack.call_multi` / `summary` semantics
beyond the shared inequality and the `admitted` telemetry flag; `run_full_attack`
untouched; the Phase 22 call-sequence contract test still passes.

---

## 4. D-03 / D-09 — aggregation (detail)

`PHASE_24_ASR_AGGREGATION_AUDIT.md`.

**Root cause.** `judge_asr.load_corpus` converted any row without
`turns`+`attack_method` into a 0-turn `Conversation`, so `failed: true` rows
inflated the denominator. Separately, `max(scores)` raised on `None` evaluator
outcomes, discarding completed conversations into that same failure path.

**Now:** `ASR = successful harmful goals / valid completed goals`, where

* *valid completed* = the conversation completed **and** produced at least one
  usable rubric score;
* infrastructure failures and invalid evaluator outcomes are counted separately
  (`excluded_infrastructure_failure`, `excluded_invalid_evaluator`,
  `excluded_no_turns`, and `run.failed_goals` / `run.invalid_evaluator_goals` in the
  manifest);
* a corpus with no valid goal exits `2` and writes no report — ASR is undefined,
  not 0 %.

Measured: 1 failure + 1 success → `asr = 1.0` (was `0.5`).
Mixed `[4, None]` → `max 4`, `n_invalid 1`, no exception (was `TypeError`).

---

## 5. D-05 — qualification alignment (detail)

`PHASE_24_QUALIFICATION_ALIGNMENT.md`.

`configs/qualification_r1_gpt4o.yaml` resolves through
`paper_protocol.resolve_paper_reproduction` to: gpt-4o ×3, 0.7/0.7/0.7, official
prompts, η = 0.001, ActorAttack + Crescendo + Opposite-day (Acronym training-only),
baseline condition, no budget cap. Probes now receive the configured temperatures
and prompt variant; `probe_actor_attack_pre_attack` validates the official
`infer_single` payload. Legacy defaults (`0.0` / `legacy`) remain for the older
harness config, whose tests still pass.

---

## 6. D-04 / D-06 / D-07 / D-08 — manifest (detail)

`PHASE_24_MANIFEST_REPRODUCIBILITY.md`.

Every §15 checklist field is now present: paper id, commit, dirty flag + status
hash, config hash, dataset sha256, goal-selection hash, configured/selected counts,
model ids + unverifiable revision, attacker condition, prompt hashes (three
templates **and** the rubric), per-role temperatures, attack set, NBF hash +
threshold, max turns, declared retry policy, JSON mode, usage paths + summary,
start/end timestamps, completion marker, completed/valid/failed/invalid counts, and
the ASR denominator.

---

## 7. Discrepancy register — Phase 23 → Phase 24

| ID | Severity | Status | Evidence |
|---|---|---|---|
| D-01 η sign | P0 | **Fixed** | boundary table; steering-off probe (0 target calls at h=0) |
| D-02 composite steering | P0 | **Fixed** | 8 steering tests + pilot-flag test |
| D-03 failed goals in ASR | P0 | **Fixed** | loader split; `asr 1.0 vs 0.5`; zero-denominator exit 2 |
| D-04 prompt hashes | P1 | **Fixed** | manifest block asserts official hashes + rubric fingerprint |
| D-05 qualification stack | P1 | **Fixed** | new config resolves; probe temperature/variant tests |
| D-06 dirty git | P1 | **Fixed** | `git_dirty` + `git_status_sha256` in manifest |
| D-07 dataset identity | P2 | **Fixed** | `dataset_provenance` (sha256, goal hash, counts) |
| D-08 run metadata | P2 | **Fixed** | `run` block (timestamps, complete, denominator, usage paths) |
| D-09 evaluator robustness | P2 | **Fixed** | `aggregate_scores`; mixed-type test |
| D-10 empty chain can score | P2 | **Unresolved (by design)** | faithful to the official `attack_single`; recorded, not changed |
| D-11 HF hub metadata requests | P2 | **Unresolved (environment)** | operational: set `HF_HUB_OFFLINE=1`; no code path changed |
| D-12 config keys not consumed | P3 | **Partially fixed** | `threshold`/`eta` equality now enforced; `protocol` / `N_retry` / `sampling` remain declarative |
| D-13 inverted docstrings | P3 | **Fixed** | two docstrings corrected |
| D-14 failure-record test gap | P3 | **Partially fixed** | consumer side covered (`load_corpus_split`); the producer path is exercised only by the offline manifest test |
| D-15 duplicate goals / mock append | P3 | **Not applicable this phase** | dataset has no duplicates; unchanged |
| D-16 upstream HarmBench byte check | P3 | **Unresolved (network)** | needs network; file hash pinned instead |

No P0 remains. See `PHASE_24_FINAL_READINESS_REPORT.md` for the classification.

---

## 8. Preservation statements

| Requirement | Evidence |
|---|---|
| NBF checkpoint / architecture / embedding / state / score | untouched; hash re-verified |
| Historical prompt variants | legacy constants untouched; legacy fingerprints still recorded |
| Frozen experiment outputs | `git status --porcelain -- results` empty |
| Frozen configurations | `configs/reproduction_*.yaml` untouched |
| Previous phase reports | not edited (Phase 21's superseded banner predates this phase) |
| Official NBF checkpoint bytes | unchanged |
| Generic Crescendo / Opposite-day behaviour | identical at η = 0 (all frozen configs); only the η→threshold boundary changed |
| ActorAttack composite sequence | unchanged; Phase 22 contract test passes |
| Research scope | no new attack, no C-1/I-1, no R2, no model substitution, no budget cap |

---

## 9. Commands

```powershell
Set-Location "C:\Users\CSLAB\Desktop\hemanth\Guardbound"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue

python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=line --junitxml=_phase24_final.xml
python -m pytest tests/test_phase24_p0_fixes.py -q
python scripts/audit_prompt_fidelity.py
python scripts/cloud_pilot.py --config configs/paper_reproduction_r1_gpt4o.yaml --variant r1_paper --dry-run --limit 2
python scripts/cloud_model_qualification.py --config configs/qualification_r1_gpt4o.yaml --dry-run
```

Result: **1131 collected / 1127 passed / 4 known environmental failures / 0 errors
/ 0 skipped**.
