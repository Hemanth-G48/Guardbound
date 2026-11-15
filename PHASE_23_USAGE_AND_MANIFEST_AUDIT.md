# PHASE 23 — USAGE, MANIFEST AND PERSISTENCE AUDIT

Observability and reproducibility of an R1 run. Evidence comes from a real
`--offline-mock` pilot run (fake transports, zero spend) plus source inspection.

Probe output: scratchpad `mock_run/` (`r1_paper.manifest.json`,
`r1_paper.jsonl`, `r1_paper.usage.json`, `r1_paper.usage.jsonl`), 1 goal ×
1 attack × 8 turns.

---

## 1. Usage telemetry (§14)

| # | Requirement | Observed | Verdict |
|---|---|---|---|
| 1 | One record per provider call | 32 usage rows for a 32-call conversation (8 turns × attacker + target + evaluator) | PASS |
| 2 | Roles distinguishable | `by_role` = attacker / target / evaluator; every row carries `role` | PASS |
| 3 | Input tokens | `input_tokens` present | PASS |
| 4 | Cached input tokens | `cached_tokens` present (0 when the provider reports none) | PASS |
| 5 | Output tokens | `output_tokens` present | PASS |
| 6 | Total tokens | `total_tokens` present | PASS |
| 7 | Duration | `duration_s` per row and aggregated | PASS |
| 8 | Request ids | `request_id` per row (null when the provider omits it) | PASS |
| 9 | Provider errors | `outcome`, `error_category`, `error_message` per row; `by_error_category` aggregate | PASS |
| 10 | Unknown usage honest | `usage_available: false`; tokens 0; cost `None`; `usage_unavailable_calls` counts them | PASS |
| 11 | Unknown cost never zero | `estimate_cost` returns `None` when the rate is absent; `cost_basis: unavailable` | PASS |
| 12 | Usage does not alter the payload | `test_usage_accounting_does_not_alter_the_provider_payload` compares request bodies with and without a ledger — identical; suite green in this phase | PASS |
| 13 | JSONL + summary persistence | `r1_paper.usage.json` (summary) and `r1_paper.usage.jsonl` (one row per call) both written | PASS |
| 14 | No request blocked by a custom cap | no `budget_usd` / `check_budget` / `remaining_usd` symbol anywhere; `grep` over `*.py` finds only tests asserting their absence and the R1 validator rejecting a budget key | PASS |

Recorded row fields (all 21): `role`, `provider`, `model`, `endpoint`,
`input_tokens`, `cached_tokens`, `output_tokens`, `total_tokens`, `duration_s`,
`request_id`, `outcome`, `error_category`, `error_message`, `estimated_cost_usd`,
`cost_basis`, `usage_available`, `temperature`, `json_format`, `finish_reason`,
`refusal`, `timestamp`.

Summary fields: `calls`, `failed_calls`, `usage_unavailable_calls`, `input_tokens`,
`cached_tokens`, `output_tokens`, `total_tokens`, `duration_s`,
`estimated_cost_usd`, `priced_calls`, `unpriced_calls`, `cost_basis`, `rate_card`,
`by_role`, `by_model`, `by_outcome`, `by_error_category`.

Mock-run totals: 32 calls, 0 failed, 0 unpriced, cost basis `rate_card` — the
numbers are harness artefacts (fake transport), not a cost estimate for R1.

---

## 2. Manifest completeness (§15)

Present in the produced manifest (30 top-level keys):

| Requirement | Field | Value |
|---|---|---|
| Git commit | `git_commit` | `89ed03f7e5d51755842e9169aa6b6e5df36b0439` |
| Configuration SHA-256 | `config.sha256` | `cfe071c5…` |
| Model identifier | `models`, `model_ids`, `providers` | `gpt-4o-2024-08-06` ×3 |
| Model revision unverifiable | `model_identity.roles.*.resolved_revision = null` + written reason | PASS |
| Attacker condition | `attacker_prompt_condition` | `baseline` |
| Prompt hashes | `attacker_prompt.system_prompt_templates` | see §3 (defect) |
| Temperature per role | `temperature`, `model_identity.roles.*.generation` | 0.7 / 0.7 / 0.7 |
| Attack set | `attacks`, `paper_protocol.attacks` | resolved list |
| NBF checkpoint hash | `nbf.sha256` (when NBF enabled) | `cea1a75b…` |
| NBF threshold / eta | `nbf.threshold`, `nbf.eta` | 0.001 |
| Max turns | `max_turns` | 8 |
| Retry policy | `protocol` (declared) | `retries: 0`, `repair/fallback: NONE` |
| JSON mode | `structured_output` | `json_object` |
| Environment | `environment` | python/platform/packages |
| Result kind | `result_kind` | `HARNESS VALIDATION ONLY …` for mock runs |

Absent, against the §15 checklist:

| Missing item | Consequence | Register |
|---|---|---|
| Paper version / id | the manifest does not name the paper it reproduces (only `methodology_label`) | D-08 (P2) |
| Dirty-git flag | the tree **is** dirty (Phases 20-22 uncommitted), so `89ed03f` does not describe the code that ran | D-06 (P1) |
| Dataset SHA-256 | the exact dataset bytes cannot be recovered from the manifest | D-07 (P2) |
| Goal-selection hash | the 200 selected behaviours cannot be reconstructed from the manifest | D-07 (P2) |
| Usage records | usage lives in sidecar files; the manifest records no path or digest | D-08 (P2) |
| Start / end timestamps | only `model_identity.execution_date_utc` (one instant, at manifest build time) | D-08 (P2) |
| Provider errors | per-goal failures are in the results JSONL only | D-08 (P2) |
| Incomplete goals | no completed/failed counter | D-08 (P2) |
| Final ASR denominator | not recorded anywhere | D-08 (P2) |

No seed or repetition count is invented: `experiment.seed` is `null` in the R1
config and the manifest records it as `null`.

Paper-specified vs code-recovered is distinguished: `paper_protocol` carries
`target_model_provenance: "paper"` and `attacker_model_provenance:
"code-recovered"`, and the R1 YAML annotates each value.

---

## 3. Manifest prompt hashes are wrong for the R1 path (D-04, P1)

The manifest records `prompt_variant: official`, but the hashes under
`attacker_prompt.system_prompt_templates` are the **legacy** ones:

| Prompt | Manifest | legacy | official (actually sent) | Describes |
|---|---|---|---|---|
| crescendo | `7df65dbd9ce4bed4…` | same | same | identical, no ambiguity |
| opposite_day | `1cd2c2d0120bc88d…` | `1cd2c2d0…` | `dfd5b4da…` | **legacy** |
| acronym | `6b8b11a5ff15f06a…` | `6b8b11a5…` | `f3afb6a9…` | **legacy** |
| rubric | *not recorded* | — | `146589a4…` | **missing** |

Root cause: `_cloud_common.arm_manifest_block` → `attacker_arms.
attacker_prompt_fingerprints()`, which hashes the module constants rather than
`prompt_variants.<prompt>(variant)`.

Consequence: a manifest-based reconstruction of an R1 run would reconstruct the
wrong prompt bytes — precisely what §15's "trace each result to its inputs" asks
for. The prompts actually sent are correct (audit `paper_path_faithful = true`);
only the record is wrong. Minimal fix: hash the resolved variant and add the
rubric hash.

---

## 4. Output persistence (§16)

| Behaviour | Implementation | Assessment |
|---|---|---|
| Successful record | one JSONL line appended per goal (`_cloud_common.append_jsonl`) | works (probe: 1 line, 21 keys) |
| Refusal response | stored verbatim in `turns[].response`; `was_filtered` flags NBF filters | works |
| Evaluator failure | score `None` stored in `rubric_scores` | stored, but see §5 |
| Malformed JSON | `AttackGenerationError` → goal fails loudly → `failed: true` record | works |
| Provider timeout | classified error → `failed: true` record + `failure_category` | works |
| Process interruption | `append_jsonl` is a plain buffered append (no fsync, no atomic rename); last line may be truncated | see §4.1 |
| Incomplete actor records | `ActorAttackRun` is returned only on success; a mid-actor exception yields no record at all | see §4.2 |
| Partially completed goals | not represented — a goal is either a record or a failure record | acceptable |
| Duplicate goal IDs | dataset has none; no runtime guard exists | D-15 (P3) |
| Re-executing a completed goal | blocked by the existing-results check (`cloud_pilot.py:385`) except under `--offline-mock` | works, no resume |

### 4.1 Truncated tail is loud, not silent

`judge_asr.load_corpus` (`scripts/judge_asr.py:68`) does
`json.loads(line)` for every non-empty line; a truncated final line raises and the
whole load fails. That is the safe direction (no silent partial data), but it means
an interrupted run needs the partial tail removed by hand. No automated repair
exists, by design (`repair: NONE`).

### 4.2 A partial run cannot be *detected* from the outputs

* The manifest is written **before** the first request (`cloud_pilot.py:441`) and
  never updated.
* It records `dataset.num_goals_requested` (post-`--limit`) but no completion count,
  no `finished_at`, and no run status.
* Therefore a results file with 47 of 600 expected records is accompanied by exactly
  the same manifest as a completed run. An operator must compare line counts
  against `attacks × goals`.

**Verdict: an incomplete experiment cannot be *reported* as complete by any field
that claims completion — but nothing marks completion either.** Severity P2 (D-08),
because the check is manual rather than machine-verifiable.

---

## 5. Failed goals are counted as attack failures by the ASR loader (D-03, P0)

Executed demonstration (probe 2, `failure_aggregation`):

Input JSONL: one pilot failure record
(`{"attack": "actor_attack", "goal": "behaviour-1", "failed": true,
"error_class": "CloudTimeoutError", …}`) and one real conversation record.

| Observation | Value |
|---|---|
| `judge_asr.load_corpus` result | 2 conversations; the failed row becomes a **0-turn** conversation |
| ASR denominator | **2** |
| Successful behaviours | 1 |
| ASR | **0.5** |
| ASR if the failure were excluded | **1.0** |

`evaluate_asr` judges every conversation it is given and `compute_asr` uses
`len(verdicts)` as the denominator, so an infrastructure failure is scored as a
failed jailbreak. Nothing warns, and the pilot's own `success` field is not
consulted by the loader.

This is the single most consequential observability defect found in this phase: at
600 conversations, a 5 % provider/contract failure rate would depress the reported
ASR by ~5 percentage points with no indication.

Minimal fix (Phase 24): in `load_corpus`, skip rows with `failed: true` (and rows
lacking `turns`), count them separately, and surface
`n_excluded_failed` in the ASR aggregate and the manifest.

---

## 6. Verdict

Usage telemetry is complete, role-attributed, budget-free and honest about unknown
costs. The manifest covers most R1 provenance (commit, config hash, models,
temperatures, attack set, NBF hash/threshold, attacker condition) but has four
gaps and one wrong field (prompt hashes). Persistence is append-only and
fail-loud, with no completion marker. The one P0 in this document is the ASR
denominator treatment of failed goals, which is an aggregation defect, not a
persistence one.
