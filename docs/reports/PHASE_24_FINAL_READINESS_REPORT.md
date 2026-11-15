# PHASE 24 — FINAL READINESS REPORT

**Final classification: `READY FOR MINIMAL QUALIFICATION`** (implementation only).

All three Phase 23 P0 defects are corrected and independently verified offline. The
qualification harness now describes the R1 stack rather than a legacy default, and
the future minimal qualification protocol is fully specified. **No paid request was
made, and this classification does not authorise one** — a separate explicit user
decision is still required.

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Are the three P0 defects fixed? | **Yes** — NBF η, ActorAttack steering, ASR denominator; each has a behavioural probe and regression tests |
| Is the R1 configuration still the approved one? | **Yes** — models, 0.7 ×3, official prompts, η 0.001, attack set, 200 goals, label 5, no budget |
| Is the manifest now self-describing? | **Yes** — paper, commit + dirty state, config hash, dataset hash + goal hash, prompt hashes incl. rubric, timestamps, completion marker, counts, ASR denominator, usage paths |
| Can the qualification gate describe R1? | **Yes** — `configs/qualification_r1_gpt4o.yaml`, probes at the R1 temperatures/variant, ActorAttack pre-attack probe |
| Do frozen artifacts remain intact? | **Yes** — `git status --porcelain` empty for `results/ runs/ checkpoints/ data/ reproduction*.yaml nbf_original_stuff` |
| Offline suite | **1131 collected / 1127 passed / 4 failed / 0 errors / 0 skipped** — the same 4 environmental failures as before |

---

## 2. Phase 22/23 independent verification, re-run

| Check | Result |
|---|---|
| Evaluator temperature 0.7/0.7/0.7 on the R1 path | PASS (11 call sites; Phase 22 audit) |
| Attack set exact, validated before any client construction | PASS |
| ActorAttack official composite sequence | PASS (call-sequence equality retained) |
| Prompt bytes on the paper path | PASS (`paper_path_faithful = true`) |
| Dataset: 200 behaviours, 0 duplicates, stable order | PASS |
| NBF checkpoint / architecture / score / determinism | PASS (hash `cea1a75b…a136fe`) |
| Usage telemetry complete, no cap | PASS |
| Frozen artifacts | PASS |

---

## 3. ActorAttack execution audit

`PHASE_24_ACTORATTACK_STEERING_AUDIT.md`. Control flow unchanged and verified:
`prepare_attack` once, fresh `[system]` conversation per actor, `N_retry = 3`,
rubric before step judge, one summary per non-skipped actor, score-5 early stop,
per-actor records, one behaviour per goal. New: the runner honours `steer_target`,
installs the official candidate filter, and refuses an impossible
`use_safety_filter` request instead of silently ignoring it.

R1 (`steer_target: false`): candidate filtering only, no steered wrapper — matching
`mode: safety_filtering` in the configuration and the author's `run.py` semantics.

---

## 4. R1 configuration trace

`PHASE_23_R1_CONFIGURATION_TRACE.md` (still accurate) plus the corrections:

| Parameter | YAML | Runtime consumer | Verified |
|---|---|---|---|
| η | `nbf.eta: 0.001` | `raw_threshold_from_eta` → `-0.001` → `set_safety_filter` / `SteeredLLMChat` | PASS |
| threshold | `nbf.threshold: 0.001` | must equal η (validator); converted once | PASS |
| steer_target | `false` | composite + generic runners | PASS |
| temperatures | 0.7 ×3 | attacks / runner / evaluator helpers | PASS |
| prompt variant | `official` | `build_attack` → templates; rubric `official_prompt` | PASS |
| budget | absent | `resolve_paper_reproduction` rejects the key | PASS |

---

## 5. Dataset integrity

200 records, sha256 `ac789de8…`, 0 duplicates, order preserved, identical set for
all three attacks, `target_system` and `max_rounds` constant per `[P]` B.1. The
manifest now records the file hash and the ordered goal-id hash
(`4a7dc375…` domain) so the exact selection is reconstructable.

---

## 6. NBF checkpoint and filtering

Checkpoint `cea1a75b…a136fe`; predictor 1536→32→32→5; dynamics 1536→512→512→768;
MPNet 768-d; zeros(768) initial state; `h = P(5) − max(P(1..4))`, deterministic and
recomputable to 1e-12. Filter rule now defined once and applied everywhere:
**filter iff `h + η ≥ 0`** (admit iff `h < −η`), with the boundary table verified for
`0.0, −0.0005, −0.001, −0.0015, −0.002, +0.001`.

---

## 7. Prompt fidelity

`paper_path_faithful = true`. Official bytes match the release for Opposite-day,
Acronym and the rubric; Crescendo, refusal and disclaimer are single-text. R1
selects `official`; historical configs select `legacy`. The manifest now records
the hashes **of the variant actually sent**, plus the rubric hash and a legacy block
for historical reconstruction.

---

## 8. Qualification harness findings

`PHASE_24_QUALIFICATION_ALIGNMENT.md`. The harness now qualifies the R1 stack:
gpt-4o ×3, 0.7/0.7/0.7, official prompts, η 0.001, ActorAttack + Crescendo +
Opposite-day (Acronym training-only), baseline condition, no budget cap. It adds an
ActorAttack pre-attack contract probe and two declared metrics; a missing probe is
`INSUFFICIENT_DATA`, never a pass. Legacy defaults remain available for the older
harness config.

---

## 9. Usage telemetry and budget removal

Unchanged and verified: one record per provider call (21 fields), roles
distinguishable, input/cached/output/total tokens, duration, request id, outcome,
error category, cost + basis, `usage_available`, cost `None` (never 0) when
unknown, JSONL + summary persistence, no cap and no blocking anywhere. The manifest
now also links the usage files and embeds the summary.

---

## 10. Manifest and crash recovery

`PHASE_24_MANIFEST_REPRODUCIBILITY.md`. The manifest carries a `run` block with
`started_at`, `finished_at`, `complete`, `goals_expected/completed`,
`valid_completed_goals`, `invalid_evaluator_goals`, `failed_goals`,
`asr_denominator`, `usage_path`, `usage_jsonl_path`, `results_path`. A partial run
has `complete: false` and a non-zero exit. A truncated JSONL tail still fails loudly
in the loader (no silent repair). Failed goals are excluded from the ASR corpus and
counted separately.

---

## 11. Offline test results

| Collected | Passed | Failed | Errors | Skipped | Time |
|---:|---:|---:|---:|---:|---:|
| **1131** | **1127** | **4** | **0** | **0** | 198.4 s |

The 4 failures are the pre-existing environmental ones (2 CUDA-gated VRAM guards,
2 missing `Qwen/Qwen3.5-4B` weights), identical node IDs to the Phase 23 baseline.
No newly introduced failure; two tests were updated because they asserted the
defects this phase removed (documented in `PHASE_24_TEST_RESULTS.md` §4).

---

## 12. Remaining discrepancy register

| ID | Severity | Status |
|---|---|---|
| D-01 | P0 | **Fixed** |
| D-02 | P0 | **Fixed** |
| D-03 | P0 | **Fixed** |
| D-04 | P1 | **Fixed** |
| D-05 | P1 | **Fixed** |
| D-06 | P1 | **Fixed** |
| D-07 | P2 | **Fixed** |
| D-08 | P2 | **Fixed** |
| D-09 | P2 | **Fixed** |
| D-10 | P2 | **Unresolved, accepted** — an empty pre-attack chain can be scored via the summary turn; this is the official `attack_single` behaviour and is faithful, not a regression. Recommended mitigation for analysis: exclude goals with zero chain queries (`actor_records[*].turns == 0` before the summary) from the ASR corpus. |
| D-11 | P2 | **Unresolved, environmental** — loading the real embedder performs HF Hub cache-metadata requests. Run with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. No OpenAI request; no spend. |
| D-12 | P3 | **Partially fixed** — `threshold`/`eta` equality enforced; `protocol`, `N_retry`, `sampling` remain declarative (enforced in code, not by the key). |
| D-13 | P3 | **Fixed** |
| D-14 | P3 | **Partially fixed** — the consumer side of the failure record is tested; the producer path is only exercised through the offline manifest test. |
| D-15 | P3 | **Not applicable** — the dataset has no duplicate goals; the append guard is unchanged by design. |
| D-16 | P3 | **Unresolved** — upstream HarmBench byte comparison needs network; the file hash is pinned instead. |

No P0 and no P1 remains.

---

## 13. Exact prerequisites before a paid minimal qualification

1. **Explicit user authorisation** for one qualification run of
   `configs/qualification_r1_gpt4o.yaml` (the paid path still requires
   `--confirm-paid-request`).
2. `OPENAI_API_KEY` exported in the shell only — never a file, never an argument.
3. Optional but recommended: `$env:HF_HUB_OFFLINE = "1"` (D-11) once the MPNet
   weights are cached, so the NBF probe cannot touch the network.
4. Nothing else: the configuration is validated, the thresholds are declared, the
   probes are implemented and the report records the stack under test.

---

## 14. Final decision

# `READY FOR MINIMAL QUALIFICATION`

Justification against the phase's own criteria:

* **All P0 defects corrected and independently verified** — D-01, D-02, D-03 each
  have a behavioural probe (boundary table, steering counters, ASR denominator
  arithmetic) plus regression tests.
* **The qualification harness matches R1** — same models, temperatures, prompt
  bytes, margin and attack set, with an ActorAttack probe; the configuration is
  validated by the same `paper_protocol` invariants as the run it gates.
* **No paper-critical mismatch remains** — the four environmental test failures and
  the residual P2/P3 items are documented, none of them changes the paper's method
  or the R1 execution path.
* **The future qualification protocol is fully specified** —
  `PHASE_23_QUALIFICATION_READINESS.md` §5 (roles, prompts, attacks, pass criteria,
  invalid-response definition, usage, error classification, safe-stop conditions),
  now backed by an implementation that can actually execute it.

This status does not authorise an API request. A separate, explicit decision is
required before the first paid qualification run, and another before any formal
experiment.
