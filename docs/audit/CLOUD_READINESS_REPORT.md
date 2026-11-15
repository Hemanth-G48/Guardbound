# CLOUD_READINESS_REPORT

> **SUPERSEDED — do not quote the numbers in this file as the current state.**
>
> This report was written before the environment repair, before the pricing and
> deprecation verification, before the model-snapshot pinning, and before the
> prompt-fidelity audit. Its test counts refer to an intermediate tree.
>
> Current documents:
> * `CLOUD_FINAL_READINESS_REPORT.md` — consolidated status
> * `CLOUD_FINAL_READINESS_RECONCILIATION.md` — authoritative test counts and the
>   A/B/C regression matrix
> * `PAPER_PROMPT_FIDELITY_AUDIT.md` — byte-level prompt comparison
> * `PAPER_METHODOLOGY_FIDELITY_AUDIT.md` — pipeline-level comparison
> * `CLOUD_EXPERIMENT_PROTOCOL_FINAL.md` — the experiment that will run
>
> Kept only as a record of the state at the time it was written.

**Status: `CLOUD READY — AWAITING API KEY`**

No paid API request was made. Cloud inference has **not** been verified against a
real endpoint — only against the official SDK over an in-process fake transport.

Date of audit: 2026-11-10 · Branch `main` · HEAD `89ed03f` (+ uncommitted work)

---

## 1. Readiness table

| Category | Status | Evidence |
|---|---|---|
| Paper methodology audit | **PASS** | Paper extracted and read (Sec. 5.1, B.1–B.2); `PAPER_METHODOLOGY_COMPARISON.md` |
| Official implementation comparison | **PASS** | `NBF-LLM` release read file-by-file: `steering.py`, `attacks/utils/{generate,check_refusal,check_disclaimer,evaluate_with_rubric}.py`, the three `run.py` files, `build_ssm_nbf.py` |
| Existing Guardbound audit | **PASS** | `CLOUD_INTEGRATION_AUDIT.md` §1–2 |
| OpenAI SDK integration | **PASS** | Official SDK 2.54.0, Chat Completions, `max_retries=0`, `tests/test_cloud_provider.py` (54) |
| Cloud provider implementation | **PASS** | `OpenAIChatLLM` + `build_cloud_role_llm`; works alongside the local provider |
| GPT-4o mini configuration | **PASS** | `configs/cloud_research_gpt4o_mini.yaml`, labelled model substitution |
| Original GPT-4o reference configuration | **PASS** | `configs/cloud_research_reference.yaml` |
| Attacker compatibility | **PASS** (offline) | Attacker-generated contract consumed by all three attacks; prompt asserted byte-identical |
| Target compatibility | **PASS** (offline) | Target calls non-JSON, temperature 0.7, dataset `target_system` seeded |
| Evaluator compatibility | **PASS** (offline) | Refusal / disclaimer / rubric at temperature 0.0; rubric failure ≠ score 1 |
| Structured output | **PASS** (offline) | 9 required contract cases + `json_object` / `json_schema` / `none` modes |
| Attack compatibility | **PASS** (offline) | Crescendo, OppositeDay, Acronym through the real runner; ActorAttack unchanged and excluded |
| NBF compatibility | **PASS** (interface) — see §3.1 | Checkpoint verified by sha256 and loads/scores; filtering proven provider-agnostic. End-to-end NBF-ON is **blocked in this environment** by a broken `transformers`/`torchvision` install |
| API key security | **PASS** | Env-only, fail-fast error, redaction on every message, `.env.example` tracked, `.gitignore` fixed |
| Token tracking | **PASS** | input/cached/output/total per call, per role and per model |
| Cost tracking | **PASS** (rates unverified) | YAML rate card, `None` not `0` when unknown, budget guard before each request |
| Offline tests | **PASS** | 120 new tests, all pass, no network, no key |
| Regression tests | **PASS** | Failure set **identical** to baseline; 878 passed / 44 failed / 4 errors |
| Paid API smoke test | **NOT RUN** | prepared, gated by `--confirm-paid-request` |
| Formal cloud experiment | **NOT RUN** | configs prepared, not executed |

---

## 2. Test evidence

### Historical claim vs. actual state

| | Claimed in the brief | Measured before my changes | Measured after |
|---|---|---|---|
| passed | 804 | **758** | **878** |
| failed | 2 | **44** | **44** |
| errors | 0 | **4** | **4** |
| total | 806 | 806 | 926 |

The historical "804 passed, 2 failures" does **not** hold in this environment.
The measured baseline was 758 / 44 / 4, and my changes add 120 passing tests
without altering the failing set:

```
before=48 after=48
Compare-Object (before, after) -> no output     # identical node-id sets
```

### Root causes of the 48 pre-existing failures

| Count | Root cause | Classification |
|---|---|---|
| 44 | `torchvision 0.22.1+cu118` is ABI-incompatible with `torch 2.14.1+cpu`, so `import transformers` raises `RuntimeError: operator torchvision::nms does not exist`. Every test that imports the official attack code (which imports `transformers`) fails at import. | **Environment defect**, unrelated to this work |
| 2 | `tests/test_phase14_2_rotation.py` — assertions about the VRAM pre-run guard (the repo's own Phase 15 note). | Pre-existing repo state |
| 2 | `tests/test_phase8_architecture.py` — `reproduction.yaml` names `Qwen/Qwen3.5-4B`, which is not cached locally; validation correctly refuses. | Pre-existing repo state / environment |

Repairing `torchvision` would restore the 44 blocked tests (including the entire
parity suite) and would also unblock NBF-ON cloud runs.

### New tests

| File | Tests | Scope |
|---|---|---|
| `tests/test_cloud_provider.py` | 54 | factory, key handling, config validation, messages, parameters, structured output, the 9 contract cases, 8 error categories, no-retry, budget |
| `tests/test_cloud_usage.py` | 32 | rate card, cost math, ledger, budget, persistence, thread safety, redaction, classification |
| `tests/test_cloud_research_integration.py` | 34 | attacks × cloud transport, prompt/temperature preservation, contract-failure separation, NBF off/on, target-call suppression, termination, config invariants, script gates |
| **Total** | **120** | all offline |

---

## 3. Conditions attached to the verdict

### 3.1 NBF-ON cannot run end-to-end on this machine yet

* The validated checkpoint is present and **sha256 matches**
  `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe`.
* It loads and scores: `h(zeros, zeros) = -0.18855…`, i.e. the barrier is not
  degenerate.
* The filtering *plumbing* is proven provider-agnostic with a deterministic stub
  barrier (accept path, filter path, target-call suppression, threshold sign).
* But the embedding model (`all-mpnet-base-v2` via `sentence-transformers`)
  cannot load because `transformers` is broken by the torchvision ABI mismatch.
  The pilot warms up the embedder **before** the first paid request, so this
  fails safe — but it means `cloud_baseline_b_nbf_on` will refuse to start until
  the environment is repaired.

**Action before an NBF-ON cloud run:**
```bash
pip install --force-reinstall torchvision   # matching torch 2.14.1+cpu
python -c "import sentence_transformers; print('ok')"
python -m pytest tests/parity -q            # should unblock 36 tests
```

### 3.2 Prices are unverified

`configs/cloud_pricing.yaml` ships with `verified: false` because
`gpt-4o` / `gpt-4o-mini` are legacy ids and were **not** present in the current
flagship price table when this was prepared. Every cost figure is labelled
`unverified_rate_card`. Verify against the official pricing page and set
`verified: true` before trusting any cost number.

### 3.3 Model ids are floating aliases

`gpt-4o` and `gpt-4o-mini` do not pin a snapshot, and the provider does not
return the resolved revision. For a reproducible run, use
`gpt-4o-2024-08-06` / `gpt-4o-mini-2024-07-18`; the manifest records whatever id
was configured.

### 3.4 gpt-4o-mini is not yet qualified

Nothing here assumes `gpt-4o-mini` behaves like `gpt-4o` for attacker usability,
evaluator agreement, or refusal behaviour. `scripts/cloud_model_qualification.py`
is the gate, and it has not been run.

---

## 4. What is proven, and what is not

**Proven (offline, deterministic, reproducible):**

* the OpenAI provider produces the request payload the reference implementation
  produces — Chat Completions, `json_object`/`text` response formats, per-call
  temperature, no output cap by default;
* a missing key fails before any network access, with an actionable message and
  no key material in it;
* eight provider failure categories are classified, recorded, and never retried;
* every call is recorded with tokens, cached tokens, duration, request id, and a
  cost estimate whose basis is labelled;
* the budget guard stops the next request once the cap is reached, without a
  network call;
* the three paper attacks run unmodified through the existing runner against a
  cloud `ChatLLM`;
* the attacker system prompt on the wire is byte-identical to the official one;
* evaluator calls use temperature 0.0 while generation uses 0.7;
* malformed attacker output raises `AttackGenerationError` (not a silent empty
  query), and an unusable evaluator reply yields `None` (never a fabricated 1);
* NBF filtering accepts/filters on the documented threshold sign and suppresses
  the target call when filtering;
* the two research variants differ only in NBF enablement, asserted before spend;
* the NBF checkpoint digest, dims and threshold are pinned and verified.

**Not proven:**

* that a real OpenAI endpoint returns what the mock returns;
* that `gpt-4o-mini` is a usable attacker / target / evaluator for this protocol;
* the actual cost of a run;
* wire-level parity with the authors' own provider logs;
* end-to-end NBF-ON execution on this machine (see §3.1).

---

## 5. Definition of success — assessment

| Requirement | Met? |
|---|---|
| Original research methodology audited | **Yes** — paper + vendored official code, quoted in the comparison document |
| Cloud provider integrated without breaking local inference | **Yes** — local provider tests pass; factory default path unchanged |
| Attack, evaluator and NBF pipelines preserve methodology | **Yes** — no attack/defense/model/evaluation file was modified; prompts asserted on the wire |
| All offline tests pass or failures fully explained | **Yes** — 120 new pass; all 48 pre-existing failures categorised with root causes |
| No paid API requests made | **Yes** — zero requests; every script is gated |
| Ready for an explicitly approved smoke test after the key arrives | **Yes** — `python scripts/cloud_smoke_test.py --confirm-paid-request` |

---

## 6. Next actions (in order)

1. **Repair `torchvision`** (unblocks 44 tests + NBF-ON).
2. **Verify `configs/cloud_pricing.yaml`**, set `verified: true`.
3. Provide `OPENAI_API_KEY` in the shell — not in a file, not in a command line.
4. `python scripts/cloud_smoke_test.py --dry-run` → expect all preconditions green.
5. `python scripts/cloud_smoke_test.py --confirm-paid-request --budget-usd 0.10`.
6. `python scripts/cloud_model_qualification.py --confirm-paid-request --budget-usd 1.00`.
7. `python scripts/cloud_pilot.py --offline-mock ...` (free harness check).
8. `python scripts/cloud_pilot.py --variant cloud_baseline_a_nbf_off --attack crescendo_paper --limit 5 --budget-usd 2.00 --confirm-paid-request`.
9. Same for `cloud_baseline_b_nbf_on`.
10. Only then the full matrix, with a set budget.

Do not begin the formal experiment until steps 4–9 have passed and been reviewed.
