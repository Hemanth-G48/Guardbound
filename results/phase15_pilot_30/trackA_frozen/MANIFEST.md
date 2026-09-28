# Phase 15 — Track A frozen 30-goal pilot: evidence manifest

## Status

**PILOT COMPLETE — 180 / 180 records, 30 / 30 goals.** This directory holds the
frozen Track A pilot evidence. Integrity gate: **ALL CHECKS PASS**.

| | |
|---|---|
| Command | `python scripts/phase14_full_reproduction.py --batch 0 --limit 30 --attack all --condition both --config configs/reproduction_phase14_frozen.yaml --structured-output-mode constrained_json` |
| Matrix | 30 goals x 3 attacks x 2 NBF conditions = **180 runs** |
| Live output path | `results/phase14/batch00.jsonl` (hardcoded in the runner; no `--out` flag) |
| Attacks | `crescendo_paper`, `opposite_day`, `acronym` (ActorAttack excluded) |
| Per-goal order | Crescendo ON/OFF, OppositeDay ON/OFF, Acronym ON/OFF |
| Attacker | `Qwen/Qwen3-4B-Instruct-2507` (pinned slot 1) |
| Target | `microsoft/Phi-4-mini-instruct` |
| Evaluator | `meta-llama/Llama-3.2-3B-Instruct` |
| Embedder | `all-mpnet-base-v2` |
| NBF checkpoint SHA256 | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` |
| Dataset SHA256 | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` |
| Config hash | `61ea203c20e68c51` (unchanged across the abort and the resume) |
| Seed base | 42 (per-run seed derived from goal id / attack / condition) |
| `filter_trials` | 3 (declared frozen value; `actor_attack` absent) |

## Files in this directory

| File | Contents |
|---|---|
| `batch00_goals000-029_180records.jsonl` | **Final 180 records** (goals 0–29). SHA256 prefix `FCB952F1598B697B033D605B0791CFBB…` — byte-identical to `results/phase14/batch00.jsonl` at completion. |
| `batch00_goals000-017_108records.jsonl` | Snapshot of the **first 108 records** (goals 0–17), taken before the resume. SHA256 prefix `C12A562A9454D83B6CC0EE93…` — byte-identical to `results/phase14/batch00.jsonl` at snapshot time; unchanged after the resume, which confirms the 72 new records were appended, never rewritten. |
| `PHASE_15_INTERIM_REPORT_108_OF_180.md` | Copy of the interim abort report (`PHASE_15_30_GOAL_PILOT_REPORT.md` at the repo root) describing the abort at 108/180 and the root-cause diagnosis. |
| `MANIFEST.md` | This file. |

The completion audit is in `PHASE_15_PILOT_COMPLETION_REPORT.md` at the repo root.

The original `results/phase14/batch00.jsonl` is **not deleted or overwritten**: the
runner's resume mechanism keys on `run_id` from that exact path, so the 108 completed
run ids must remain readable there for the resume to skip them.

## Why the pilot stopped at 108/180

`ModelManager.pre_run_check` read `torch.cuda.mem_get_info()[0]` directly — the raw
driver-level free value, which counts PyTorch's **reusable cached** blocks as consumed.
At the start of `ON_crescendo_018` that read 4.04 GB free while 11.789 GB sat in the
reusable cache (effective availability 15.829 GB), so the 6.0 GB per-run floor tripped
and `VRAMGuardError` aborted the batch.

Classification: **INFRASTRUCTURE_FAILURE** (a false-positive measurement in this
project's own guard) — not an attack, model, parser or data failure. All 108 records
passed every integrity check, and effective availability never fell below ~9.4 GB.

## The fix applied before resuming

`src/guardbound/llm/model_manager.py` → `ModelManager.pre_run_check` now releases
reusable CUDA cache **before** measuring free VRAM, using the same sequence
`_load_with_guards` already used:

```
gc.collect() -> torch.cuda.synchronize() -> torch.cuda.empty_cache() -> torch.cuda.synchronize()
```

Thresholds (pre-load 3.0 GB, post-load 1.5 GB, per-run 6.0 GB, ceiling 22.0 GB
reserved), the pre-load gate, the post-load verification and the residency policy are
all unchanged. See `PHASE_15_PILOT_COMPLETION_REPORT.md` at the repo root for the
full audit.

## Provenance note

`worktree_sha256` is a hash of `git diff HEAD` plus `git status --porcelain`. The
infrastructure fix above changes it, so the resumed 72 records carry a different
`worktree_sha256` than the first 108. Every other identity field
(`code_commit`, `config_hash`, `checkpoint_sha256`, `dataset_sha256`, model ids,
dtype, quantization, `max_rounds`, `threshold`, `filter_trials`) is identical between
the two halves of the run.
