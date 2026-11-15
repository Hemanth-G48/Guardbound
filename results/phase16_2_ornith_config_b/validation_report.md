# Phase 16.2 — Ornith-1.5-9B Configuration B Focused Validation

**Date:** 2026-09-28
**Type:** qualification only. **No ASR experiment and no 90-run pilot were launched.**

---

## 1. Objective

Re-run Phase 16.1's Configuration B (Ornith-1.5-9B, `enable_thinking=False`,
unconstrained structured output) and determine:

1. whether its behaviour **reproduces**, and
2. whether its raw outputs are actually **accepted by the existing frozen JSON
   interface** — without extraction, fence-stripping, whitespace/quote/key repair,
   trailing-comma repair, regex extraction, retries, regeneration or fallback
   parsing of any kind.

## 2. Exact configuration

| | |
|---|---|
| Model | `ornith-ai/Ornith-1.5-9B` |
| Revision | `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` |
| Precision / quantization | bfloat16 / none |
| Backend | existing native `Ornith15ChatLLM` (`AutoProcessor` + `AutoModelForMultimodalLM`) |
| `enable_thinking` | **False** (`chat_template_kwargs = {enable_thinking: false}`) |
| Structured output | **unconstrained** (`structured_output_mode = None`) |
| Temperature / max_new_tokens | 0.7 / 256 |
| Prompts / schema / stopping | unchanged (frozen attack prompts, frozen parser) |
| Residency | sequential, one model resident |
| Seeds | `derive_seed(goal_id, "crescendo", "off")` (+ turn index in Stage B) — identical to Phase 16.1 |
| Goals | Stage A 0–19, Stage B 0–7, same order as Phase 16.1 |
| Target feedback | same replay mechanism (Track A NBF-OFF Crescendo recordings; fixed fallback for goal 1) |
| Hardware | NVIDIA RTX 4500 Ada Generation, 24.57 GB |
| Code state | `code_commit b46d6fd7…`, `worktree_sha256 662e7e94…` |
| Timestamp | 2026-09-28T19:25:58Z |

Token counts are not exposed by the frozen interface; adding them would require
backend changes, which this phase prohibits, so raw character length is recorded
instead (measurement durations and VRAM are recorded per generation).

## 3. Phase 16.1 comparison

| Measurement | Phase 16.1 Config B | Phase 16.2 | Match |
|---|---:|---:|:--:|
| Stage A valid JSON | 17 / 20 | **17 / 20** | yes |
| Stage B successful generations | 15 / 19 | **15 / 19** | yes |
| Stage B context failures | 4 | **4** | yes |
| Stage B full 3-turn sequences | 4 / 8 | **4 / 8** | yes |
| Raw output text | — | **byte-identical on all 39 generations** | yes |
| Per-goal Stage A outcomes | — | 20/20 identical | yes |

The reproduction is *exact*, including which goals failed. Two runs of this phase
(run 1 preserved as `run1_*`) also produced byte-identical output text (20/20 and
19/19), so the model + harness path is deterministic under fixed seeds.

**Interpretation caveat, stated plainly:** because the seeds and every generation
parameter are identical by design (the task requires paired, comparable runs),
this is a *reproduction of the same measurement under bit-identical conditions*
rather than an independent statistical replication. It demonstrates determinism
and confirms the earlier numbers; it does not add new sampling evidence.

## 4. Stage A results (20 isolated generations)

| Metric | Value |
|---|---:|
| Valid JSON (accepted by the frozen interface) | **17 / 20** |
| JSON failures | **3 / 20** |
| Extra-text rate | 0.15 (3/20) |
| Empty rate | 0.00 (0/20) |
| Reasoning-only rate | 0.00 (0/20) |
| Accepted via direct `json.loads` | **17** |
| Accepted via the frozen structural extraction | **0** |
| Mean latency | 5.65 s |
| Median latency | 5.58 s |
| Peak VRAM | 19.963 GB |

## 5. Stage B results (8 sequences, up to 3 turns each)

| Metric | Value |
|---|---:|
| Successful generations / attempted | **15 / 19** (0.79) |
| Context failures | **4** (2 JSON_PARSE_ERROR, 2 EXTRA_TEXT) |
| Completed 3-turn sequences | **4 / 8** |
| JSON failures | 2 (both truncated) |
| Extra-text failures | 2 (both trailing comma) |
| Empty outputs | 0 |
| Reasoning-only outputs | 0 |
| Accepted via direct `json.loads` | **15** |
| Accepted via the frozen structural extraction | **0** |
| Goal preserved in the system message | True (all generated turns) |
| Role ordering correct | **True (19/19 turns)** |
| Mean latency | 7.51 s |
| Median latency | 7.09 s |

Verified message structure, from the messages actually sent to the backend:

```text
turn 1: [system, user]
turn 2: [system, user, assistant, user]
turn 3: [system, user, assistant, user, assistant, user]
```

The previous attacker query stays an `assistant` message and the replayed target
reply stays inside the existing "The last response was: …" user message. No
alternative history format was constructed.

## 6. Parser compatibility analysis (the point of this phase)

* **Every accepted reply was accepted by direct `json.loads` of the raw model
  output.** The frozen structural extraction (`_extract_json_block`) — which
  Qwen also shares — was **never needed** in either stage (0/32 accepted replies).
  So compatibility here does **not** rest on prose-extraction leniency.
* The three Stage A replies that contained a markdown fence were **rejected**, not
  rescued: inside the fence the JSON carried a trailing comma (`,}`), which the
  frozen parser correctly refuses. Nothing was cleaned to make them pass.
* No reasoning wrapper appeared at all: 0/39 replies contained a thinking marker,
  consistent with `enable_thinking=False` on this model.
* No reply was repaired, retried, regenerated or replaced; the rejection classes
  below are the frozen interface's own verdicts.

## 7. Failure taxonomy

Using the existing categories (no new ones invented):

**Stage A (20):** `VALID_JSON` 17 · `EXTRA_TEXT` 3 · `JSON_PARSE_ERROR` 0 ·
`EMPTY_OUTPUT` 0 · `REASONING_ONLY` 0 · `MISSING_REQUIRED_FIELD` 0 ·
`WRONG_FIELD_TYPE` 0
— the 3 `EXTRA_TEXT` replies are fenced JSON whose defect is a trailing comma.

**Stage B (19):** `VALID_JSON` 15 · `JSON_PARSE_ERROR` 2 · `EXTRA_TEXT` 2 ·
`CONTEXT_FAILURE` 4 (the sequence-level effect of those 4 turn failures) ·
`EMPTY_OUTPUT` 0 · `REASONING_ONLY` 0 · `MISSING_REQUIRED_FIELD` 0 ·
`WRONG_FIELD_TYPE` 0

Infrastructure classes: `MODEL_LOAD_FAILURE` 0 · `CUDA_OOM` 0 ·
`OTHER_INFRA_FAILURE` 0.

Defect shapes: trailing comma 3 (A) + 2 (B); truncation 2 (B). Complete raw
outputs are preserved verbatim for all 39 generations in `stage_a.jsonl` /
`stage_b.jsonl`.

## 8. GPU / residency

| Metric | Value |
|---|---:|
| Max resident model count | **1** |
| Allocated VRAM after load | 18.82 GB |
| Reserved VRAM after load | 18.837 GB |
| Peak VRAM | 19.963 GB |
| Model load time | 15.87 s |
| Model unload time | 0.20 s |
| Load refusals | 0 |
| CUDA OOM | 0 |
| CUDA errors | 0 |
| Allocated after unload | 0.009 GB (reserved 0.021 GB) — no leak |
| Residents after unload | 0 |

Sequential residency behaved exactly as in Phase 16.1:
`load Ornith → generate → release → (target) → …`. No guard was weakened, nothing
was quantized, no offloading, no concurrent residency.

## 9. Reproducibility assessment

Decision criteria (declared before the run, not fitted afterwards): Stage A valid
generations within ±2 of 17/20; Stage B generated turns within ±3 of 15/19;
completed 3-turn sequences within ±1 of 4/8.

```text
Stage A valid        17/20 vs 17/20   (±2)  match
Stage B generated    15/19 vs 15/19   (±3)  match
3-turn sequences      4/8  vs  4/8    (±1)  match
```

All three criteria match, and the raw text is byte-identical to Phase 16.1's
Configuration B on all 39 generations.

## 10. Integrity checks

* Track A results unchanged: `results/phase14/batch00.jsonl` and
  `results/phase15_pilot_30/trackA_frozen/batch00_goals000-029_180records.jsonl`
  both `FCB952F1598B697B033D605B0791CFBB…`; the pre-resume snapshot
  `C12A562A9454D83B6CC0EE93…`.
* Track A configuration unchanged: `configs/reproduction_phase14_frozen.yaml`
  `27B660A7052054521A94E3430013D986…`, `config_hash 61ea203c20e68c51`.
* Track A code, NBF, target, evaluator and attack algorithms: untouched (unchanged
  mtimes; no source file other than the new diagnostic script was added).
* Phase 16.1 results were not overwritten — this phase wrote only to
  `results/phase16_2_ornith_config_b/`.

## 11. Qualification decision

```text
REPRODUCED
```

Configuration B's behaviour reproduces exactly (metrics and raw text) and its
outputs are accepted by the **unmodified frozen interface** in the large
majority of generations — with a consistent, reproducible rejection rate on a
small set of specific goals.

## 12. ASR / pilot statement

No ASR, attack-success rate or success percentage was calculated from this
phase. No attack-success experiment was run. The 90-run pilot was **not**
launched, and no configuration was tuned, retried or optimised as a result of
these measurements.

## 13. Files added or modified

| Path | Status |
|---|---|
| `scripts/validate_ornith15_config_b.py` | **added** (this phase's harness) |
| `results/phase16_2_ornith_config_b/stage_a.jsonl`, `stage_b.jsonl`, `summary.json`, `validation_report.md` | **added** (evidence) |
| `results/phase16_2_ornith_config_b/run1_stage_a.jsonl`, `run1_stage_b.jsonl`, `run1_summary.json`, `run2.log` | **added** (run-1 preservation for the determinism check) |

No production, backend, parser, prompt, attack, NBF, target or evaluator file was
modified.

## 14. Conclusion — the specific question

> **Does Ornith-1.5-9B with `enable_thinking=False` and unconstrained generation
> reliably produce outputs accepted by the existing frozen JSON interface across
> isolated and multi-turn generation?**

**Partially — the behaviour is reproducible, but it is not reliable.** Measured:
17/20 isolated generations (85 %) and 15/19 multi-turn turns (79 %) were accepted
by the unmodified frozen interface, and every acceptance came from a direct JSON
parse rather than from the interface's prose-extraction leniency. But 3/20 and
4/19 generations were rejected by that same interface — reproducibly, on the same
goals — because of a trailing comma in fenced JSON or budget truncation, and 4 of
8 multi-turn sequences therefore stopped before their third turn.

A higher raw JSON percentage is **not** treated here as qualification: the
criterion was reproducible compatibility with the actual frozen interface, and
that interface rejects roughly one in five Configuration B generations.

```text
PHASE 16.2 COMPLETE — CONFIGURATION B REPRODUCED —
FROZEN-INTERFACE COMPATIBILITY PARTIAL — NO ASR STUDY — 90-RUN PILOT NOT LAUNCHED
```
