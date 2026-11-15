# PHASE 24 — MANIFEST REPRODUCIBILITY (D-04, D-06, D-07, D-08)

**Status: FIXED.** An R1 manifest now records the prompt bytes actually sent, the
state of the tree that produced them, the dataset's identity and selection, and
the completion facts needed to distinguish a 200-goal run from a partial one.

Evidence: a real `--offline-mock` pilot run (fake transports, zero spend) in a
temporary directory, asserted by
`tests/test_phase24_p0_fixes.py::test_offline_pilot_manifest_is_self_describing`.

---

## 1. D-04 — prompt hashes described the wrong bytes

**Root cause.** `_cloud_common.arm_manifest_block` called
`attacker_arms.attacker_prompt_fingerprints(condition)`, which imported the legacy
module constants. With `prompt_variant: official`, the manifest recorded
`opposite_day = 1cd2c2d0…` while the run sent `dfd5b4da…`; the rubric prompt was
not hashed at all.

**Change.**

* `prompt_variants.system_prompt_templates(variant)` and
  `prompt_variants.rubric_template(variant)` are the single resolution point;
  `prompt_variants.fingerprint(text)` produces `{sha256, chars}`.
* `attacker_prompt_fingerprints(arm, prompt_variant=None)` now emits:

```json
{
  "condition": "baseline",
  "prompt_variant": "official",
  "system_prompt_templates": {"crescendo": {...}, "opposite_day": {...}, "acronym": {...}},
  "rubric_prompt": {"sha256": "...", "chars": 3374, "fixture": {"query": "Q", "response": "R"},
                    "variant": "official"},
  "legacy_system_prompt_templates": {...},
  "evaluator_receives_instruction": false,
  "target_receives_instruction": false
}
```

* `arm_manifest_block(cfg)` passes `prompt_variant_of(cfg)`.
* The legacy block is always present, so a historical run stays reconstructable
  from an official-prompt manifest (and vice versa).

**Verified.** For R1: `prompt_variant == "official"` and
`system_prompt_templates.opposite_day.sha256 == sha256(official bytes)`; the
legacy block equals the legacy fingerprints; `rubric_prompt.sha256` equals the
hashed rubric template. The qualification report carries the same four
fingerprints.

---

## 2. D-06 — commit recorded, dirty tree not

**Root cause.** `git_commit()` was `git rev-parse HEAD` only. The working tree
contains uncommitted work, so `89ed03f` did not identify the code that ran.

**Change.** `_cloud_common.git_dirty()` and `git_status_sha256()` (SHA-256 of
`git status --porcelain`); both tolerate a missing git. The manifest now records:

```json
"git_commit": "89ed03f7e5d51755842e9169aa6b6e5df36b0439",
"git_dirty": true,
"git_status_sha256": "…64 hex…"
```

The same two fields are recorded in the qualification report.

---

## 3. D-07 — dataset identity and selection

**Root cause.** The block was
`{"path", "num_goals_requested": len(goals), "sampling"}`, where
`num_goals_requested` was the **post-`--limit`** length (1 for a 200-goal config run
with `--limit 1`), and no hash was recorded.

**Change.** `_cloud_common.dataset_provenance(cfg, goals)`:

```json
{
  "name": "HarmBench behaviours (official NBF-LLM test split)",
  "path": ".../harmbench_tasks.json",
  "sha256": "ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb",
  "num_goals_configured": 200,
  "num_goals_selected": 1,
  "goal_id_field": "task",
  "goal_id_hash": "…64 hex over the ordered goal strings…",
  "sampling": "file_order",
  "shuffle": false
}
```

The manifest also records `"limit": args.limit`, so a pilot slice is explicit
rather than implied. `--limit` can no longer make the requested count wrong.

---

## 4. D-08 — completion, timestamps and the denominator

**Root cause.** The manifest was written once, before the loop, and never updated:
no start/end time, no completion marker, no counts, no ASR denominator.

**Change.** The manifest gains a `run` block, written at start and **rewritten**
after the loop:

```json
"run": {
  "started_at": "2026-11-10T15:16:32Z",
  "finished_at": "2026-11-10T15:17:04Z",
  "complete": true,
  "goals_expected": 200,
  "goals_completed": 200,
  "valid_completed_goals": 198,
  "invalid_evaluator_goals": 1,
  "failed_goals": 1,
  "asr_denominator": 198,
  "asr_denominator_definition": "valid_for_asr records: completed conversations with at least one usable rubric score. Infrastructure failures and invalid evaluator outcomes are excluded.",
  "usage_path": ".../r1_paper.usage.json",
  "usage_jsonl_path": ".../r1_paper.usage.jsonl",
  "results_path": ".../r1_paper.jsonl",
  "usage_summary": {"calls": 32, "estimated_cost_usd": 0.0144,
                    "cost_basis": "rate_card", "unpriced_calls": 0}
}
```

`complete` is true only when `goals_completed + failed_goals == goals_expected`
(`len(attacks) × len(goals)`). The process exits non-zero when any goal failed, any
evaluator outcome was invalid, or the run is incomplete, and prints

```
complete           True (200/200 goals, 1 failed, 1 invalid evaluator)
ASR denominator    198 valid completed goals
```

The manifest also now records the paper (`arXiv:2503.00187v3`, title, authors,
venue) so a report names the work it reproduces.

---

## 5. Fields the §15 checklist requires

| Field | Present | Source |
|---|---|---|
| Paper version / id | ✅ | `manifest.paper` |
| Git commit | ✅ | `manifest.git_commit` |
| Dirty git state | ✅ | `manifest.git_dirty`, `git_status_sha256` |
| Configuration SHA-256 | ✅ | `manifest.config.sha256` |
| Dataset SHA-256 | ✅ | `manifest.dataset.sha256` |
| Goal-selection hash | ✅ | `manifest.dataset.goal_id_hash` |
| Model identifier | ✅ | `models`, `model_ids`, `model_identity` |
| Model revision as unverifiable | ✅ | `model_identity.roles[*].resolved_revision = null` + reason |
| Attacker condition | ✅ | `attacker_prompt_condition` |
| Prompt hashes (incl. rubric) | ✅ | `attacker_prompt.system_prompt_templates`, `.rubric_prompt` |
| Temperature per role | ✅ | `temperature`, `model_identity…generation` |
| Attack set | ✅ | `attacks`, `paper_protocol.attacks` |
| NBF checkpoint hash / threshold | ✅ | `nbf.sha256`, `nbf.threshold`, `nbf.eta` |
| Maximum turns | ✅ | `max_turns` |
| Retry policy | ✅ (declared) | `protocol` — enforcement is `max_retries=0` in the client (D-12, P3) |
| JSON mode | ✅ | `structured_output` |
| Usage records | ✅ | `run.usage_path`, `run.usage_jsonl_path`, `run.usage_summary` |
| Start / end timestamps | ✅ | `run.started_at`, `run.finished_at` |
| Provider errors | ✅ (per goal) | results JSONL: `failed`, `error_class`, `failure_category`; `run.failed_goals` |
| Incomplete goals | ✅ | `run.goals_completed`, `run.failed_goals`, `run.complete` |
| Final ASR denominator | ✅ | `run.asr_denominator` |

No seed or repetition count is invented: `experiment.seed` stays `null` and the
manifest records it as `null`.

Paper-specified vs code-recovered remains explicit in `paper_protocol`
(`target_model_provenance: "paper"`, `attacker_model_provenance: "code-recovered"`)
and in the R1 YAML annotations.

---

## 6. Historical manifests

New fields are additive. No existing manifest file was read, modified or
regenerated, and no historical run is rewritten: `git status --porcelain` is empty
for `results/`. A manifest produced by an older script version simply lacks the new
keys.

---

## 7. Regression coverage

```
test_manifest_prompt_hashes_describe_the_bytes_sent
test_arm_manifest_block_uses_the_resolved_variant
test_dataset_provenance_records_hashes_and_counts
test_dataset_provenance_separates_configured_from_selected
test_git_state_helpers_are_honest_about_dirty_trees
test_paper_reference_is_recorded
test_offline_pilot_manifest_is_self_describing   (end-to-end, asserts every field above)
```
