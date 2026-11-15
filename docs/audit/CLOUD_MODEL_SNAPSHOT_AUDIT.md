# CLOUD_MODEL_SNAPSHOT_AUDIT

Pinning exact model identities so a re-run cannot silently land on a different
revision. Verified 2026-11-10 against the official model documentation.

---

## 1. Why this matters here specifically

The project's reproducibility contract records model ids in manifests. A floating
alias (`gpt-4o-mini`) is **not** a stable identity: the provider may repoint it,
and the repository already contains evidence of that mechanism working — the
`gpt-3.5-turbo-0125` snapshot was shut down on 2026-10-23 while its alias family
continued to exist.

---

## 2. Aliases audited → snapshot resolution

| Alias | Default snapshot (official) | Snapshot status | Announced shutdown |
|---|---|---|---|
| `gpt-4o` | `gpt-4o-2024-08-06` | Deprecated | 2027-01-14 |
| `gpt-4o-mini` | `gpt-4o-mini-2024-07-18` | Deprecated | 2027-01-14 |

Other GPT-4o snapshots, for context from the model page: `gpt-4o-2024-11-20`
(Deprecated), `gpt-4o-2024-05-13` (**already shut down** 2026-10-23).

Sources:
`https://developers.openai.com/api/docs/models/gpt-4o`,
`https://developers.openai.com/api/docs/models/gpt-4o-mini`,
`https://developers.openai.com/api/docs/deprecations`.

### Feature compatibility with Guardbound's usage

| Requirement | `gpt-4o-2024-08-06` | `gpt-4o-mini-2024-07-18` |
|---|---|---|
| `v1/chat/completions` | Supported | Supported |
| Structured Outputs (`json_schema`) | Supported | Supported |
| JSON mode (`json_object`) | Supported | Supported |
| Context window | 128,000 | 128,000 |
| Max output tokens | 16,384 | 16,384 |
| Temperature accepted | yes | yes |

Both support everything the paper's protocol needs on the endpoint Guardbound
uses. **A dated snapshot was not assumed to be available** — the ids are read
from the model pages, which list them explicitly as (deprecated) snapshots.

---

## 3. Every place a cloud model id is configured

| Location | Kind | Change |
|---|---|---|
| `configs/cloud_research_gpt4o_mini.yaml` → `models.{attacker,target,evaluator}` | runnable | `gpt-4o-mini` → **`gpt-4o-mini-2024-07-18`** |
| `configs/cloud_research_reference.yaml` → `models.{attacker,target,evaluator}` | runnable | `gpt-4o` → **`gpt-4o-2024-08-06`** |
| `configs/cloud_qualification.yaml` → `models.*` | runnable | already `gpt-4o-mini-2024-07-18` |
| `configs/cloud_models.yaml` → `paper_reference`, `local_research_adaptation`, `mixed_*`, `paper_judge_local_target` | catalog | all four blocks pinned to snapshots |
| `scripts/cloud_smoke_test.py` → `--model` default | runnable | `gpt-4o-mini` → **`gpt-4o-mini-2024-07-18`** |
| `configs/cloud_pricing.yaml` | pricing | entries for both snapshots + both families |
| `src/guardbound/llm/openai_client.py` | code | no default; `model_id` is required — nothing to pin |
| `configs/default.yaml`, `configs/reproduction*.yaml` (`paper_reference:` blocks) | **historical documentation** | **not changed** — these record what the paper/official code used, and rewriting them would falsify the historical record |
| `tests/cloud_fakes.py` (`TEST_MODEL = "gpt-4o-mini"`) | test fixture | not changed — it is a fake transport; the family alias also exercises the rate card's exact-match path |

No script silently defaults to a cloud model: the qualification and pilot scripts
read the model from a config file, and the smoke test's default is now a snapshot.

---

## 4. Identity recorded in every manifest

`scripts/cloud_pilot.py` writes `model_identity`, and the qualification harness
writes the same block. Per role:

```json
{
  "role": "evaluator",
  "configured_model_id": "gpt-4o-mini-2024-07-18",
  "provider": "openai",
  "endpoint": "/v1/chat/completions",
  "is_dated_snapshot": true,
  "floating_alias": false,
  "snapshot_id": "gpt-4o-mini-2024-07-18",
  "resolved_revision": null,
  "resolved_revision_note": "The provider does not return the resolved model
     revision in the chat-completion response, so the served revision is not
     independently verifiable from the API result.",
  "generation": {
    "temperature_used_by_calls": {
      "attacker_generation": 0.7,
      "target_completion": 0.7,
      "evaluator_calls": 0.0
    },
    "structured_output_mode": "json_object",
    "max_output_tokens": null
  }
}
```

Plus, at the manifest top level:

```json
{
  "execution_date_utc": "2026-11-10T16:51:26Z",
  "all_roles_share_one_model": true,
  "deprecated": true,
  "announced_shutdown_date": "2027-01-14",
  "identity_source": "https://developers.openai.com/api/docs/models/gpt-4o-mini"
}
```

* **All three roles are recorded independently**, and
  `all_roles_share_one_model` makes an accidental shared id visible instead of
  implied.
* **`resolved_revision` is `null` with a written reason** — the limitation is
  stated, not hidden. The configured id is the reproducibility handle.
* **`execution_date_utc`** exists because the models have a shutdown date; the
  run date is part of the provenance.

---

## 5. Tests guarding this

`tests/test_cloud_pricing_and_snapshots.py`:

* every runnable cloud config pins a dated snapshot (regex `-\d{4}-\d{2}-\d{2}$`)
  for all three roles — a future edit back to a floating alias fails the suite;
* every pinned snapshot has a verified rate in the rate card;
* Configuration B is the mini snapshot and its substitution label contains
  "SUBSTITUTION"; Configuration A is the paper snapshot;
* the smoke test's default model is a snapshot;
* `model_identity` records each role separately, flags a floating alias, states
  the resolved-revision limitation, and records per-role generation parameters;
* a real pilot manifest (offline mock) contains `model_identity`,
  `is_dated_snapshot: true`, the shutdown date and an execution timestamp.

---

## 6. Final configuration

| Configuration | Attacker | Target | Evaluator |
|---|---|---|---|
| A — paper reference | `gpt-4o-2024-08-06` | `gpt-4o-2024-08-06` | `gpt-4o-2024-08-06` |
| B — initial cloud experiment | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` |
| Qualification | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` |
| Smoke test default | `gpt-4o-mini-2024-07-18` | — | — |

---

## 7. Limitations

1. **Both snapshots are Deprecated**, with shutdown on 2027-01-14. Snapshot
   pinning buys exact identity, not permanence. Record the run date; do not plan
   a re-run past that date.
2. **The API does not expose the served revision.** Pinning is a request-side
   guarantee; it cannot prove what the server executed.
3. **No substitution was performed.** `gpt-6-luna` is the provider's recommended
   replacement but is *not* configured anywhere — adopting it is a new model
   substitution that requires its own label, pricing entry and qualification.
