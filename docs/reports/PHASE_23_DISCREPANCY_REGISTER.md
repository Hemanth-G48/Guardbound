# PHASE 23 — DISCREPANCY REGISTER

Every entry is backed by an executed probe or a source line. No speculative
defects are listed as confirmed findings; where a finding is an inference from the
official code rather than a measurement, that is stated.

Severity scale: **P0** blocks a paper-faithful reproduction · **P1** important
methodology mismatch · **P2** reproducibility/reporting weakness · **P3**
non-critical improvement.

Counts: **3 × P0**, **3 × P1**, **5 × P2**, **5 × P3**.

---

## P0 — block paper-faithful reproduction

### D-01 — NBF η sign is inverted in the attack-level filter

| Field | Value |
|---|---|
| Area | NBF filtering (Crescendo, Opposite-day; also ActorAttack's internal path) |
| Expected — paper | filter the query iff `h(x,u) + η ≥ 0`, i.e. allow iff `h < −η` (p9 §5.2; Eq. in Sec. 5, η = 1e-3) |
| Expected — official code | `threshold = -args.threshold` (`attacks/actor_attack/run.py:211`, `crescendomation/run.py:27`, `opposite_day/run.py:26`), then `if score < threshold: accept` |
| Current — Guardbound | `if ssm_nbf_score < safety_threshold: accept` with `safety_threshold = nbf.threshold = +0.001` (`attacks/runner.py:716`, `attacks/actor_attack.py:969/1013`, wired at `runner.py:567-582`) → allow iff `h < +0.001` |
| Severity | **P0** |
| Evidence | Deterministic barrier with uniform logits → `h = 0.0`; R1 threshold `+0.001` → **3 target calls** (all admitted); official sign `−0.001` → **0 target calls**; `SteeredLLMChat` with `eta=0.001` → filtered. `PHASE_23_NBF_INTEGRITY_REPORT.md` §4. |
| Why it was invisible | every frozen local run used η = 0, where `+0` and `−0` coincide |
| Recommended action | negate the margin on the attack-level path (`set_safety_filter(..., -abs(threshold))` and `-eta` at `runner.py:706`), add regression tests at `h = 0` (expect filtered) and `h = −2η` (expect admitted). Phase 24, offline. |

### D-02 — ActorAttack composite ignores `steer_target` and cannot install the attack-level filter

| Field | Value |
|---|---|
| Area | ActorAttack NBF wiring |
| Expected | R1 sets `nbf.steer_target: false` + `mode: safety_filtering`; the official path filters candidate queries inside `call_multi`/`summary` and calls the target plainly |
| Current | `run_actor_attack_composite` (`attacks/runner.py:1201-1226`) has **no `steer_target` parameter**; it wraps the target in `SteeredLLMChat` whenever `barrier` and `embed_fn` are present (`:1251`), and never calls `attack.set_safety_filter(...)`, so `call_multi`'s guard (`actor_attack.py:953`) is always false |
| Severity | **P0** |
| Evidence | `TypeError: run_actor_attack_composite() got an unexpected keyword argument 'steer_target'`; after a barrier-enabled run: `steered_chat_constructed=true`, `attack._embed_fn=None`, `attack._use_safety_filter=False`, `attack._safety_threshold=0.0`. With `use_safety_filter=True` the filter is a **silent no-op**. |
| Consequence | within one R1 run, ActorAttack is target-steered while Crescendo/Opposite-day are candidate-filtered; neither the config nor the official semantics is honoured |
| Recommended action | add `steer_target: bool = True` and `use_safety_filter` plumbing; mirror the generic runner (`use_steered = use_safety_filter and steer_target`); call `set_safety_filter` with `−η`; test both branches with a fake barrier. Phase 24. |

### D-03 — failed goals are counted as attack failures in the ASR denominator

| Field | Value |
|---|---|
| Area | Result aggregation |
| Expected | a goal lost to a provider/contract failure is excluded from ASR and reported separately (the paper's denominator is behaviours *attacked*) |
| Current | `cloud_pilot.py:469-479` writes `{"failed": true, …}` with **no** `turns`; `scripts/judge_asr.py:70-90` converts any such row into a **0-turn `Conversation`**; `evaluate_asr` judges it and `compute_asr` counts it in the denominator |
| Severity | **P0** |
| Evidence | 1 failure + 1 successful conversation → denominator 2, successful 1, **ASR 0.5**; excluding the failure → **ASR 1.0**. `PHASE_23_USAGE_AND_MANIFEST_AUDIT.md` §5. |
| Scale | at 600 conversations, a 5 % failure rate depresses ASR by ≈5 points with no warning |
| Recommended action | skip `failed: true` / turn-less rows in `load_corpus`, expose `n_excluded_failed`, record the denominator in the ASR aggregate and the manifest. Phase 24, offline. |

---

## P1 — important methodology mismatch

### D-04 — manifest prompt hashes describe the legacy bytes; rubric hash missing

| Field | Value |
|---|---|
| Area | Manifest / reproducibility |
| Expected | the hashes recorded in a manifest identify the bytes actually sent |
| Current | `prompt_variant: official` is recorded, but `attacker_prompt.system_prompt_templates` hashes come from `attacker_arms.attacker_prompt_fingerprints()` (module constants, legacy): opposite_day `1cd2c2d0…` vs sent `dfd5b4da…`; acronym `6b8b11a5…` vs sent `f3afb6a9…`. The rubric prompt is not hashed at all. |
| Severity | **P1** |
| Evidence | `PHASE_23_USAGE_AND_MANIFEST_AUDIT.md` §3 (manifest vs `prompt_variants` computed in the same process) |
| Recommended action | hash `prompt_variants.<prompt>(resolved_variant)` and add `rubric`. Phase 24, offline. |

### D-05 — the qualification harness tests a stack R1 does not run

| Field | Value |
|---|---|
| Area | Qualification gate |
| Expected | qualify the R1 stack: gpt-4o, 0.7/0.7/0.7, official prompts, η = 0.001, including ActorAttack |
| Current | defaults to `gpt-4o-mini`; evaluator probes call the helpers with **no** temperature (legacy 0.0) at `cloud_model_qualification.py:485, 597, 641, 670, 697`; the target recall probe uses `temperature=0.0` (`:546`); prompts are the legacy constants; `nbf.threshold: 0.0`; no ActorAttack probe |
| Severity | **P1** |
| Evidence | `PHASE_23_QUALIFICATION_READINESS.md` §2-3 (call sites + config values) |
| Recommended action | one `configs/qualification_r1_gpt4o.yaml` + pass per-role temperatures/variant into the probes + one ActorAttack pre-attack contract probe. ~4 files, offline. Phase 24. |

### D-06 — manifest records a commit but not a dirty working tree

| Field | Value |
|---|---|
| Area | Reproducibility |
| Expected | the manifest identifies the code that ran |
| Current | `git_commit()` is `git rev-parse HEAD` only (`_cloud_common.py:651-661`); the tree contains 23 uncommitted Phase 20-22 paths, so `89ed03f` does not identify the executed code |
| Severity | **P1** |
| Evidence | `git status --porcelain` at audit time; manifest `git_commit = 89ed03f7…` |
| Recommended action | record `git_dirty: true/false` plus a hash of `git status --porcelain` (or commit before a paid run). Phase 24. |

---

## P2 — reproducibility / reporting weakness

### D-07 — manifest dataset block lacks SHA-256 and the goal-selection hash; `num_goals_requested` is post-`--limit`

Evidence: manifest block is `{"path", "num_goals_requested": 1, "sampling"}` for a
200-goal config run with `--limit 1`. Fix: add `sha256`, `goal_id_hash`, and both
`num_goals_configured` and `num_goals_selected`.

### D-08 — manifest has no start/end timestamps, completion marker, incomplete-goal count or ASR denominator

Evidence: 30 top-level keys, none of them temporal or completion-related; the
manifest is written before the run and never updated (`cloud_pilot.py:441`), so a
partial run is indistinguishable from a complete one. Fix: write a run-status
footer (or a `*.status.json`) with `started_at`, `finished_at`, `goals_completed`,
`goals_failed`, `excluded_from_asr`, `asr_denominator`, `usage_paths`.

### D-09 — `run_one` raises on a `None` rubric score, discarding a completed conversation

Evidence: `_scores = [None, None]` → `max(scores)` raises `TypeError`
(`cloud_pilot.py:221-254`); the goal becomes a `failed: true` record, which then
feeds D-03. Fix: filter `None` (count them as evaluator-contract failures) before
aggregating.

### D-10 — an empty pre-attack query chain can score a goal as successful

Evidence: `actors[0].queries == []` → `call_multi` returns `[system]` →
`summary` runs and scored 5 in the probe → `goal_achieved=True` with no attack
query. Faithful to the official `attack_single`, so not a regression. Fix:
flag/reject an empty chain as a contract failure (documented deviation) or record
`summary_only_success` so the analysis can exclude it.

### D-11 — loading the real embedder is not network-silent

Evidence: one probe load of `all-mpnet-base-v2` produced `HEAD`/`GET` requests to
`huggingface.co` (cache metadata; 404 on `2_Normalize/config.json`). No OpenAI
request, no spend. Fix: set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` (or
`local_files_only=True`) for strict-offline runs, including inside
`guardbound.embeddings.get_embed_fn`.

---

## P3 — non-critical

**D-12 — `experiment.protocol`, `attacks.N_retry`, `dataset.sampling/shuffle` are
recorded but not consumed.** Enforcement lives elsewhere (`max_retries=0` in
`openai_client.py:223`; `N_retry = 3` hard-coded; `load_goals` always slices).
A YAML edit to any of them would be silently ignored. Fix: assert the declared
values equal the enforced ones at startup, or drop the keys.

**D-13 — inverted docstring.** `attacks/runner.py:618` says "Returns score where
lower = more unsafe (needs filtering)". The score is `P(5) − max(P(1..4))`:
**higher** means more unsafe. This inversion is a plausible origin of D-01.

**D-14 — no test covers the pilot's failure record path.** `grep` over `tests/`
finds no `error_class` / `failure_category` / `failed: true` assertion, so the
record shape that D-03 consumes is untested.

**D-15 — no runtime guard against duplicate goal identifiers**, and
`--offline-mock` bypasses the "refuse to append to existing results" check
(`cloud_pilot.py:385`). The shipped dataset has no duplicates, so this is latent.

**D-16 — dataset provenance is not byte-verified against upstream HarmBench.**
The file is verified as the release's 200-behaviour split (count, schema, order),
not as a match to the HarmBench repository.

---

## Verified with no finding (recorded so the register is not read as exhaustive negatives)

| Area | Evidence |
|---|---|
| Evaluator/attacker/target temperatures on the R1 path | 11 call sites, all 0.7; legacy 0.0 preserved without a paper config |
| Attack set exactness and pre-flight validation | 9 rejection cases + resolved set |
| ActorAttack control flow | 7 scenario classes, target-sequence equality with `run_full_attack` |
| Generic runner untouched | `git diff` shows zero deletions in `runner.py` |
| Prompt bytes on the paper path | `paper_path_faithful = true` |
| NBF checkpoint / architecture / score / determinism | hash, dims, `h` recomputation to 1e-12 |
| Dataset 200 behaviours | count, uniqueness, order, shared set across attacks |
| Usage telemetry | 21 fields/row, 3 roles, cost basis, no cap, payload unchanged |
| Budget removal | no `budget_usd` / `check_budget` / `remaining_usd` anywhere outside tests asserting absence |
| Frozen artifacts | `git status` empty for `results/ runs/ checkpoints/ data/ reproduction*.yaml` |
| Test suite | 1067 / 1063 / 4 / 0, identical failure set to Phase 22 |

---

## Ordering for Phase 24

1. D-01 (three lines + two tests) and D-02 (parameter + plumbing + tests) — both are
   NBF correctness and belong in one change.
2. D-03 (ASR loader) — otherwise every future ASR number is suspect.
3. D-04, D-06, D-07, D-08 (manifest) — one pass over the manifest builder.
4. D-05 (qualification config + probe temperatures).
5. D-09 … D-16 as housekeeping.

Nothing in this list touches a frozen experiment, the checkpoint, the attack
algorithms, the evaluator rubrics or the approved R1 decisions.
