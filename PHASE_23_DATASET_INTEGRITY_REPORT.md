# PHASE 23 — DATASET INTEGRITY REPORT

HarmBench selection used by R1. All checks executed offline against the file named
in the configuration; nothing was downloaded, added, removed or reordered.

Machine-readable manifest:
`results/audit/phase23_r1_dataset_manifest.json`.

---

## 1. What R1 uses

| Item | Value |
|---|---|
| Config key | `dataset.path` |
| File | `nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json` |
| SHA-256 | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` |
| Records in file | **200** |
| `dataset.num_goals` | 200 |
| Goals resolved by `load_goals` | **200** |
| Fields | `task`, `target_system`, `max_rounds` (no other keys) |

Provenance: **[P]** the paper's main evaluation uses 200 HarmBench behaviours
(p9 §5.1; Appendix B.1 gives the 6-token target system prompt and `max_rounds = 8`).
**[C]** the file itself ships inside the author's release as its HarmBench test
split. **[G]** Guardbound reads it through `_cloud_common.load_goals`.

---

## 2. Integrity checks (§10)

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Correct HarmBench dataset present | PASS | file exists; 200 records; the three fields the loader expects |
| 2 | Exactly 200 intended behaviours selected | PASS | `records_in_file = 200`, `resolved_goals = 200`, no truncation (`unused_records = 0`) |
| 3 | No unintended duplicates | PASS | 0 duplicate `task` values (200 distinct strings) |
| 4 | No behaviour silently dropped | PASS | selection is `cases[:200]` over a 200-element list |
| 5 | Goal identifiers stable | PASS | identity is the `task` string; goal-id hash `4a7dc375…` over the ordered list |
| 6 | Attack-to-goal mapping correct | PASS | `cloud_pilot.py:448-449` iterates `for attack in attacks: for case in goals`, so all three attacks see the same 200 cases in the same order |
| 7 | Same behaviour set for the three attacks | PASS | same `goals` list object for every attack in the run |
| 8 | Original ordering preserved | PASS | `[c["task"] for c in goals] == tasks[:200]` |
| 9 | Provenance documented | PASS (with a stated gap, §4) | see manifest `provenance` block |
| 10 | Contamination filtering traceable | PASS | none applied; neither the paper nor the release specifies one |

Additional observations:

* `target_system` is identical for all 200 records: `You are a helpful assistant.`
  — matching **[P]** B.1's "6-token" system prompt.
* `max_rounds` is 8 for all 200 records — matching **[P]** B.1 and
  `attacks.max_turns: 8`.
* The dataset carries **no numeric IDs**; the goal string is the only identifier.
  Two behaviours that differed only in wording would be distinct goals, which is
  correct, but it means "goal ID" in any analysis must be handled as free text
  (and hashed, as above, for manifests).

---

## 3. Selection procedure (`scripts/_cloud_common.py:455-474`)

```python
cases = json.load(open(path))
num_goals = dataset_cfg.get("num_goals") or len(cases)
selected = cases[: int(num_goals)]          # file order, no shuffle, no filter
if limit is not None:
    selected = selected[: int(limit)]       # pilot --limit
```

Deterministic: the same file always yields the same ordered prefix. There is no
resume/skip logic, so a re-run reproduces the same 200-goal set.

---

## 4. Provenance gap (stated, not hidden)

The file is verified as **the release's own 200-behaviour test split** — count,
schema, per-record constants and ordering all check out. It is **not** verified as
a byte match to the upstream HarmBench repository release, because no upstream
checksum is available offline and this phase forbids downloading data.

Consequence: if the authors had silently altered a behaviour string relative to
upstream HarmBench, this audit would not detect it. The manifest pins the file's
SHA-256 (`ac789de8…`), so the exact bytes used by any R1 run are recoverable and
comparable later.

**Severity: P3** (provenance limitation, not a blocker; register **D-16**).
Recommendation: when network access is authorised, record the upstream HarmBench
release tag and the corresponding file hash in the same manifest.

---

## 5. Manifest weakness found while auditing this (§15)

The pilot's `dataset` block records:

```json
{"path": ".../harmbench_tasks.json", "num_goals_requested": 1, "sampling": "file_order"}
```

for a run invoked with `--limit 1`. Two problems:

1. `num_goals_requested` is `len(goals)` **after** `--limit`, so it reports 1, not
   200. The name implies the configured target.
2. Neither the file SHA-256 nor the goal-id hash is recorded.

A manifest-based reconstruction therefore cannot prove which 200 behaviours ran.
Register entry **D-07 (P2)**; the values are given above so a Phase 24 fix is a
two-line change (`sha256_of(dataset.path)`, hash of the ordered `task` list).

---

## 6. Verdict

**Dataset integrity: PASS** for the 200-behaviour selection (count, uniqueness,
order, schema, per-attack identity). Provenance is documented with one explicit
limitation (no upstream byte comparison). The only defect in this area is the
manifest's incomplete dataset block, which is a reporting issue, not a data issue.
