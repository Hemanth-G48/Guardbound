# PHASE 23 — FINAL READINESS REPORT

**Final classification: `NOT READY`.**

Three P0 defects block a paper-faithful R1 reproduction. None of them is caused by
the Phase 22 work being absent or sloppy: two are NBF wiring defects that were
latent while η = 0, and one is an aggregation defect in a script. All three are
bounded, offline-fixable, and mapped to exact lines in
`PHASE_23_DISCREPANCY_REGISTER.md`.

No paid request, qualification, pilot or experiment was executed. `OPENAI_API_KEY`
was unset for the whole phase. No budget cap was introduced. No frozen artifact was
modified.

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Are the Phase 22 changes real? | **Yes** — temperatures, attack set, composite runner, η value, official prompt bytes, budget removal and NBF preservation all verify independently. |
| Is the ActorAttack control flow faithful? | **Yes** — call-sequence equality with the official loop in every scenario tested. |
| Is the R1 configuration internally consistent? | **Almost** — `nbf.threshold` and `nbf.steer_target` do not reach the code with the semantics the paper requires (D-01, D-02). |
| Is the measured ASR trustworthy? | **Not yet** — provider/contract failures are counted as attack failures (D-03). |
| Are the prompts right? | **Yes on the wire**, **no in the manifest** (D-04 records the legacy hashes). |
| Is the dataset right? | **Yes** — 200 behaviours, no duplicates, stable order, same set for all three attacks. |
| Is the NBF right? | **Checkpoint/architecture/score yes; the η decision no** (D-01). |
| Can a minimal qualification run start? | **Not yet** — the harness would qualify a different stack (D-05), and D-01/D-02 would make the defended ActorAttack arm non-faithful anyway. |
| Test suite | 1067 collected / 1063 passed / 4 failed / 0 errors — identical failure set to Phase 22; the failures are environmental and were not touched. |

Deliverables: nine documents (this one + eight companions), plus a machine-readable
dataset manifest at `results/audit/phase23_r1_dataset_manifest.json`.

---

## 2. Phase 22 independent verification

`PHASE_23_INDEPENDENT_IMPLEMENTATION_AUDIT.md`.

| Item | Verdict | Basis |
|---|---|---|
| Evaluator/attacker/target temperature 0.7 on the R1 path | PASS | 11 recorded calls, role attributed by client identity, all 0.7; payload carries `temperature: 0.7` |
| Legacy 0.0 default without a paper config | PASS | observed rubric/refusal at 0.0 |
| Attack set exact + pre-flight | PASS | 9 rejection cases; validation before any client construction |
| ActorAttack composite control flow | PASS | scenario matrix, §3 below |
| Crescendo/Opposite-day generic path untouched | PASS | `git diff` on `runner.py`: **zero deletions** |
| η value 0.001 propagated | PASS (value) | config → `nbf_settings` → `run_one` → runner/steered |
| η decision semantics | **FAIL** | D-01 |
| Prompt bytes on the paper path | PASS | `paper_path_faithful = true` |
| Legacy prompt bytes preserved | PASS | legacy rubric hash unchanged (`4b5504a5…`) |
| Budget cap / enforcement absent | PASS | no symbol surfaces outside tests asserting absence; R1 rejects a budget key |
| NBF checkpoint/architecture/score | PASS | hash `cea1a75b…a136fe`, 1536-32-32-5 / 1536-512-512-768, `h` recomputation to 1e-12 |
| Frozen artifacts | PASS | `git status --porcelain -- results runs checkpoints data …` empty |

---

## 3. ActorAttack execution audit

`PHASE_23_ACTORATTACK_CONTRACT_AUDIT.md`.

The official sequence (`run_actor_attack:208` → `attack_single:189` →
`call_multi:78` → `summary:135`) was transcribed and re-executed with fresh
scripted doubles. Guardbound matched the official **target-call sequence** in every
scenario: single-actor success, all-actors-processed, all-fail, `unknown` skip,
persistent rejection (4 attempts), empty chain, multi-query chains, summary <5,
summary =5, and two-goal repetition.

Also verified: `prepare_attack` once; fresh `[system]` conversation per actor;
`N_retry = 3`; rubric before step judge; one summary per non-skipped actor;
score-5 early stop; per-actor records separated; one behaviour per goal.

Two wiring defects, both P0:

* **D-02** — the composite runner has **no `steer_target` parameter**
  (`runner.py:1201`), so R1's `steer_target: false` cannot be honoured; it always
  installs `SteeredLLMChat` when a barrier is present, and never calls
  `set_safety_filter`, so the official candidate filter is unreachable (and a
  `use_safety_filter=True` argument would be a silent no-op).
* **D-01** (shared with Crescendo/OD) — the attack-level filter compares
  `h < +threshold` while the paper and the release use `h < −η`.

Edge-case verdicts: an exception is **loud** (pilot records `failed: true`,
continues, exits 1) rather than silently corrupting a goal; an invalid rubric reply
becomes `None` and is later discarded by the pilot's aggregation (D-09); an empty
query chain can score a goal successful — faithful to the official code, recorded
as D-10.

---

## 4. R1 configuration trace

`PHASE_23_R1_CONFIGURATION_TRACE.md`.

Every paper-critical value has a YAML → runtime path: models, three temperatures,
η, max turns, 200 goals, unsafe label 5, system prompt, prompt variant, enabled and
training-only attacks, JSON mode, retry-at-provider, checkpoint hash, dims, no
budget. Two exceptions: the threshold **sign** (D-01) and the composite's missing
`steer_target` (D-02).

Values the paper does not specify are not invented: `seed: null`, no repetitions,
no token limits, no CIs.

One "declared but not enforced" weakness: `experiment.protocol` is recorded in the
manifest but no code reads it (D-12); enforcement is `max_retries=0` and the absence
of repair/fallback branches.

---

## 5. Dataset integrity

`PHASE_23_DATASET_INTEGRITY_REPORT.md`.

| Check | Result |
|---|---|
| File | `nbf_original_stuff/…/data/test/harmbench_tasks.json` |
| SHA-256 | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` |
| Records / selected | 200 / 200 |
| Duplicates | 0 |
| Missing task field | 0 |
| Goal-id hash | `4a7dc3757902cbacb84a21fedcadcfb674747d3d07edec8d62019321bbc60206` |
| Ordering | preserved (`cases[:200]`) |
| Same set for all attacks | yes (`for attack in attacks: for case in goals`) |
| `target_system` | `You are a helpful assistant.` ×200 **[P] B.1** |
| `max_rounds` | 8 ×200 **[P] B.1** |
| Contamination filter | none applied; none specified by the paper or release |

Provenance is documented with one honest limitation: the file is verified as the
release's 200-behaviour split, not as a byte match to upstream HarmBench (D-16).

---

## 6. NBF checkpoint and filtering

`PHASE_23_NBF_INTEGRITY_REPORT.md`.

| Item | Value | Verdict |
|---|---|---|
| Checkpoint SHA-256 | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` | PASS |
| Predictor | 1536 → 32 → 32 → 5 | PASS |
| Dynamics | 1536 → 512 → 512 → 768 (f_θ, g_θ) | PASS |
| Embedding | MPNet, `(1, 768)` | PASS |
| Initial state | zeros(768) | PASS |
| Score | `P(5) − max(P(1..4))`, recomputed to 1e-12, deterministic | PASS |
| R1 η value | 0.001 in config, resolved, and passed to `SteeredLLMChat` | PASS |
| Filter rule — `SteeredLLMChat` | filter iff `h + η ≥ 0` | PASS (paper-faithful) |
| Filter rule — attack level | allow iff `h < +0.001` (should be `h < −0.001`) | **FAIL (D-01)** |

Executed demonstration: barrier score exactly `0.0`, η = 0.001 →
`+0.001` threshold admits the query (3 target calls), `−0.001` filters it
(0 calls), `SteeredLLMChat` filters it. The defect lives in the comparison
operand, not in the checkpoint or the score, and is invisible at η = 0 — which is
why every frozen local experiment is unaffected.

---

## 7. Prompt fidelity

`PHASE_23_INDEPENDENT_IMPLEMENTATION_AUDIT.md` §6. `scripts/audit_prompt_fidelity.py`
re-run: `paper_path_faithful = true`.

| Prompt | Paper path | Matches release |
|---|---|---|
| Crescendo attacker (all rounds) | one text | yes |
| Opposite-day attacker | `dfd5b4da0a580630…` | yes |
| Acronym attacker | `f3afb6a9c9fb6c20…` | yes |
| Rubric | `146589a4fba6d42c…` | yes |
| Refusal (2 copies), disclaimer | one text each | yes |
| Target conversation shape | dataset `target_system` + alternating turns | yes |
| Provider payload | messages unchanged, `json_object`, temperature preserved | yes |

Legacy renders remain different by design (disclosed, hashed, needed for frozen
reproduction). R1 selects `official`; historical configs resolve to `legacy`.

Manifest defect: the recorded hashes are the **legacy** ones (D-04).

---

## 8. Qualification harness findings

`PHASE_23_QUALIFICATION_READINESS.md`.

The harness cannot qualify the R1 stack as shipped: it defaults to
`gpt-4o-mini`, calls the evaluator helpers without a temperature (legacy 0.0), uses
the legacy prompt bytes, sets `nbf.threshold: 0.0`, and has no ActorAttack probe.
Ten of the fifteen checklist items fail or are partial (D-05, P1).

Minimal correction proposed (not implemented): one
`configs/qualification_r1_gpt4o.yaml`, pass the configured per-role temperatures and
prompt variant into the probes, set η = 0.001, add one ActorAttack pre-attack
contract probe — about four files, fully offline.

A complete future qualification protocol (roles, prompts, attacks, pass criteria,
invalid-response definition, usage, error classification, safe-stop conditions) is
specified in that document. It contains no spending cap, and it assumes nothing
about model availability: a future authorised run is what verifies the model.

---

## 9. Usage telemetry and budget removal

`PHASE_23_USAGE_AND_MANIFEST_AUDIT.md` §1.

All 14 requirements pass: one record per call; roles distinguishable; input /
cached / output / total tokens; duration; request ids; provider errors with
categories; `usage_available`; unknown cost `None` (never 0); payload unchanged by
the ledger; JSONL + summary persistence; no cap anywhere.

Observed in the offline-mock run: 32 usage rows across attacker/target/evaluator,
21 fields per row, cost basis `rate_card`, 0 unpriced, 0 failed. Those numbers are
harness artefacts, not an R1 cost estimate.

---

## 10. Manifest and crash recovery

`PHASE_23_USAGE_AND_MANIFEST_AUDIT.md` §2-§5.

Present: git commit, config SHA-256, models, `model_identity` with
`resolved_revision: null` + reason, attacker condition, temperatures, attack set,
NBF hash/threshold (when enabled), max turns, protocol block, JSON mode,
environment, result kind, `paper_protocol` (paper vs code-recovered provenance).

Missing: dirty-git flag (D-06), dataset SHA-256 and goal-selection hash (D-07),
start/end timestamps, completion marker, incomplete-goal count, ASR denominator,
usage paths (D-08). Paper version/id is not recorded (folded into D-08).

Wrong: prompt hashes (D-04).

Persistence: append-only JSONL, no fsync/atomic rename (a truncated tail fails
loudly in the ASR loader), "refuse to append to existing results" guard, no resume.
A partial run is accompanied by the same manifest as a complete one — completion
must be checked by line count.

**The single most consequential observability finding:** failed goals become
0-turn conversations in the ASR corpus and inflate the denominator (D-03, P0).

---

## 11. Full offline test results

```
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=no --junitxml=_phase23_baseline.xml
```

| Collected | Passed | Failed | Errors | Skipped | Time |
|---:|---:|---:|---:|---:|---:|
| **1067** | **1063** | **4** | **0** | **0** | 172.4 s |

Identical failure set to the Phase 22 baseline, node for node:

```
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured
tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid
tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present
```

Two require CUDA (torch is CPU-only), two require `Qwen/Qwen3.5-4B` weights that are
not on disk. They were **not** modified or skipped. Newly introduced failures: none.

Focused runs: `test_phase22_paper_fidelity.py` 62 passed;
`test_cloud_nbf_integration.py` 7 passed (real checkpoint + real embedder);
prompt-fidelity audit `paper_path_faithful = true`.

A green-ish suite is not the reason for any verdict in this report: the three P0
defects are all covered by offline probes, not by test failures.

---

## 12. Remaining discrepancy register

`PHASE_23_DISCREPANCY_REGISTER.md`. Summary:

| ID | Severity | Area | One-line |
|---|---|---|---|
| D-01 | **P0** | NBF | attack-level filter inverts η (`h < +0.001` instead of `h < −0.001`) |
| D-02 | **P0** | ActorAttack | composite runner has no `steer_target` and cannot install the official candidate filter |
| D-03 | **P0** | Aggregation | failed goals counted in the ASR denominator (0-turn conversations) |
| D-04 | P1 | Manifest | recorded prompt hashes are the legacy bytes; rubric hash missing |
| D-05 | P1 | Qualification | harness tests mini/0.0/legacy/η=0 and omits ActorAttack |
| D-06 | P1 | Manifest | commit recorded, dirty tree not |
| D-07 | P2 | Manifest | no dataset SHA-256 / goal hash; `num_goals_requested` is post-`--limit` |
| D-08 | P2 | Manifest | no start/end, completion marker, incomplete-goal count, ASR denominator |
| D-09 | P2 | Aggregation | `max(scores)` raises on `None` → completed conversation discarded |
| D-10 | P2 | ActorAttack | empty query chain can score a goal successful (faithful to official) |
| D-11 | P2 | Environment | real embedder load performs HF Hub metadata requests |
| D-12 | P3 | Config | `protocol` / `N_retry` / `sampling` recorded but not consumed |
| D-13 | P3 | Docs | inverted score docstring in `runner.py:618` |
| D-14 | P3 | Tests | no test for the pilot's `failed: true` record |
| D-15 | P3 | Persistence | no duplicate-goal guard; mock bypasses the append guard |
| D-16 | P3 | Dataset | provenance not byte-verified against upstream HarmBench |

---

## 13. Exact blockers before a minimal qualification run

1. **D-01** — make the attack-level filter use `−η`; regression tests at
   `h = 0` (filtered) and `h = −2η` (admitted). Otherwise every NBF-ON
   Crescendo/Opposite-day ASR is produced by the wrong decision rule.
2. **D-02** — give `run_actor_attack_composite` a `steer_target` parameter, wire
   `use_safety_filter` + `set_safety_filter`, and select the mechanism from the R1
   config. Otherwise ActorAttack's defended arm is non-faithful and inconsistent
   with the other two attacks in the same run.
3. **D-03** — exclude `failed: true` rows from the ASR corpus and report them
   separately. Without this, a 600-conversation run's ASR silently absorbs every
   infrastructure failure.
4. **D-05** — correct the qualification configuration/probes, otherwise the
   qualification verdict would describe a stack R1 never runs.

D-04, D-06–D-08 should also be fixed before a paid run so the manifest is
self-describing; they do not block the *decision* to run but they do block the
ability to reconstruct the run afterwards.

None of these requires a paid request. Items 1–3 are small, local, offline changes;
item 4 is configuration plus probe plumbing.

---

## 14. Final readiness classification

# `NOT READY`

Reason: unresolved P0 issues of exactly the kinds the phase definition enumerates —
**incorrect threshold** (D-01), **ActorAttack composite mismatch** (D-02) and
**incomplete result aggregation** (D-03). The R1 implementation is otherwise
structurally correct: the attack set, temperatures, prompt bytes, dataset,
checkpoint, architecture and usage telemetry all verify, and the regression suite is
unchanged.

What would change the verdict: D-01, D-02 and D-03 corrected and re-audited offline,
D-05 corrected so the qualification gate describes R1, and a fresh independent pass
over the same probes.

**This classification does not authorise an API request.** Even at
`READY FOR MINIMAL QUALIFICATION`, a paid qualification run would still require a
separate, explicit user decision.

---

### Provenance labels used in this report

* **[P]** paper-specified — arXiv:2503.00187v3, with page/section/table given at the
  point of use.
* **[C]** code-recovered — taken from the author's release because the paper is
  silent (e.g. the attacker model, `N_retry = 3`, `C_refused < 10`, the
  `threshold = -args.threshold` convention).
* **[G]** Guardbound implementation observation — measured in this phase.
* **[A]** user-approved decision — the R1 table in the phase brief.
* **[Assumption]** stated where something could not be measured offline.
