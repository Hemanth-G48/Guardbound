# CLOUD_INTEGRATION_AUDIT

What was inspected, what was changed, what was deliberately left alone, and why.
Companion documents: `PAPER_METHODOLOGY_COMPARISON.md` (component-by-component
fidelity), `CLOUD_MODEL_CONFIGURATION.md`, `CLOUD_SETUP_GUIDE.md`,
`CLOUD_READINESS_REPORT.md`.

**No paid API request was made at any point during this work.** All validation
runs against the real OpenAI SDK over an in-process fake HTTP transport.

---

## 1. What was inspected

### Original research material

| Source | Path | Used for |
|---|---|---|
| Paper text | `2503.00187v3.pdf` (extracted to text; Sec. 5.1, Appendix B.1–B.2) | model list, temperatures, rounds, success criterion, NBF setup, refusal rule |
| Official entry point | `nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/steering.py` | default attacker/target models, threshold sign convention, output file format |
| Official generation | `.../attacks/utils/generate.py` | exact OpenAI call: Chat Completions, `response_format`, temperature, no `max_tokens`, per-model branches |
| Official evaluators | `.../attacks/utils/check_refusal.py`, `check_disclaimer.py`, `evaluate_with_rubric.py` | evaluator prompts, `temperature=0.0`, thresholds, return contracts |
| Official attacks | `.../attacks/crescendomation/run.py`, `opposite_day/run.py`, `acronym/run.py` | loop structure, backtracking, disclaimer OR, candidate filtering |
| Official NBF | `.../attacks/build_ssm_nbf.py`, `train.py` | barrier score formula, state transition, checkpoint architecture |
| Official dataset | `.../data/test/harmbench_tasks.json` | 200 cases, `max_rounds = 8`, `target_system` present on all 200 |

### Current repository

`src/guardbound/llm/{base,openai_client,provider_factory,local_client,model_manager}.py`,
`src/guardbound/attacks/*`, `src/guardbound/defense/steered_chat.py`,
`src/guardbound/models/{dynamics,predictor,compat}.py`,
`src/guardbound/evaluation/*`, `configs/reproduction_phase14_frozen.yaml`,
`scripts/run_reproduction.py`, `scripts/run_attacks.py`, `tests/*`.

### Key findings from that inspection

1. A cloud provider **already existed** (`OpenAIChatLLM` + `provider_factory`),
   but it was a thin transport: no usage/cost capture, no structured-output
   control, no error taxonomy, no budget guard, and a silent
   `response_format` omission for non-JSON calls (the reference sends
   `{"type":"text"}` explicitly).
2. `provider_factory.build_role_llm` already resolved each role independently,
   so mixed local/cloud stacks needed **no** new architecture.
3. `OpenAIChatLLM` let the SDK's own retry loop run (`openai.OpenAI()` default
   `max_retries=2`), which contradicts the formal protocol `retries = 0`.
4. `.env.example` was itself git-ignored by the `.env.*` rule.
5. `tests/` lacked `__init__.py` while `tests/parity/*` imports
   `tests.parity.utils`, so the suite only collected on machines without a
   `tests` package in site-packages.

---

## 2. What was changed

### 2.1 `src/guardbound/llm/openai_client.py` — extended in place

Same class, same constructor prefix (`model_id`, `api_key_env`), same
`generate()` contract. Added, all keyword-only:

| Addition | Why |
|---|---|
| `max_retries=0` on the SDK client | Enforces the formal protocol; the SDK cannot retry behind our back. |
| `role` + shared `ledger` | Per-role cost attribution. |
| `structured_output` / `json_schema` / `json_schema_name` | Native Structured Outputs as an explicit opt-in; JSON mode remains the default. |
| `max_output_tokens`, `top_p`, `timeout_s`, `base_url`, `organization`, `project` | Explicit local→cloud parameter mapping, reported not assumed. |
| `client=` injection | Offline tests need no key and no network; the factory's fail-fast key check is bypassed only when a transport is supplied. |
| `build_request()` + `parameter_map()` | The exact payload and the mapping table are inspectable without a network call. |
| `last_request_stats` | Mirrors the local backend's `last_generation_stats`; carries tokens, cached tokens, duration, request id, finish reason, refusal flag, outcome, cost. |
| Classified exceptions | `CloudAuthError`, `CloudRateLimitError`, `CloudTimeoutError`, … instead of raw SDK errors; secrets redacted in every message. |
| Refusal / truncation / empty / content-filter outcomes | Recorded as data, never "repaired". |
| Missing-key `CloudAuthError` | Clear, actionable, and contains no key material. |

Response format sent:

| Caller asks | JSON mode (default) |
|---|---|
| `json_format=True` | `{"type": "json_object"}` |
| `json_format=False` | `{"type": "text"}` (reference parity) |

Evidence for choosing **Chat Completions over the Responses API**: the official
reference calls `client.chat.completions.create`, and the project-wide `ChatLLM`
contract is a `messages` list — Chat Completions accepts that directly, so no
translation layer was needed. Structured Outputs and JSON mode are both
documented as available on Chat Completions for `gpt-4o-mini` and
`gpt-4o-2024-08-06` and later.

### 2.2 `src/guardbound/llm/cloud_errors.py` — new

Failure taxonomy (`authentication`, `permission`, `rate_limit`, `timeout`,
`connection`, `server`, `invalid_request`, `context_length`, `content_filter`,
`budget_exhausted`, `provider_error`), a classifier that works with the SDK
installed *or* with a class-name-only test double, and the redaction helper
(`redact`, `describe_api_key`) used by every log line and exception.

### 2.3 `src/guardbound/llm/usage.py` — new

`ModelRate` / `RateCard` loaded from YAML (never hardcoded), `estimate_cost`,
`UsageRecord`, thread-safe `UsageLedger` with `by_role` / `by_model` /
`by_outcome` / `by_error_category` breakdowns, and `check_budget()` which raises
`BudgetExceededError` **before** a request is sent.

Two deliberate refusals to guess:

* a model with no rate yields `estimated_cost_usd = None` and
  `cost_basis = "unavailable"` — never 0;
* a configured budget with an unrated model raises rather than running blind.

### 2.4 `src/guardbound/llm/provider_factory.py` — extended

`build_cloud_role_llm()` factored out; `build_role_llm(..., ledger=...)` and
`build_three_model_stack(..., ledger=...)` pass it through; `normalize_provider()`
exposed so scripts share the factory's local/cloud decision.
`validate_cloud_role_config()` **rejects** local-only keys on a cloud role
(`top_k`, `do_sample`, `chat_template_kwargs`, `quantization`, `device_map`,
`revision`, `dtype`, `residency`, `parameters`, …) instead of dropping them.

### 2.5 Scripts — new

| Script | Sends requests? | Gate |
|---|---|---|
| `scripts/cloud_smoke_test.py` | yes (2 tiny calls) | `--confirm-paid-request`; `--dry-run` for zero spend |
| `scripts/cloud_pilot.py` | yes | `--confirm-paid-request`; `--dry-run`; `--offline-mock` |
| `scripts/cloud_model_qualification.py` | yes | `--confirm-paid-request`; `--dry-run` |
| `scripts/_cloud_common.py` | no | helpers only |

None is imported by application code, none runs at import time, and none is
invoked by the test suite.

### 2.6 Configs — new

`configs/cloud_pricing.yaml`, `configs/cloud_models.yaml`,
`configs/cloud_research_reference.yaml`, `configs/cloud_research_gpt4o_mini.yaml`.
No existing config was modified.

### 2.7 Tests — new / adjusted

* **New:** `tests/cloud_fakes.py`, `tests/test_cloud_provider.py` (54 tests),
  `tests/test_cloud_usage.py` (32), `tests/test_cloud_research_integration.py`
  (34) — 120 tests, all offline.
* **`tests/conftest.py`** — added the `fake_backend` fixture.
* **`tests/__init__.py`** — new, empty-but-for-a-docstring. Makes `tests` a
  regular package so `from tests.parity.utils import ...` resolves to this
  directory instead of a same-named package in site-packages.
* **`tests/test_llm_clients_mocked.py`** — two tests previously asserted that a
  call succeeds with **no** API key configured. That is precisely the behaviour
  the audit required changing (clear missing-key error), so they now set an
  obvious placeholder (`test-key-not-a-credential`) that is only ever handed to
  a fake module. The missing-key behaviour is asserted separately in
  `tests/test_cloud_provider.py`.

### 2.8 Security files

* `.gitignore` — added `!.env.example` (it was being ignored by `.env.*`).
* `.env.example` — rewritten: `OPENAI_API_KEY=` with an empty value, a note
  that the key is read from the environment only, and an explicit warning
  against committing/pasting/logging it.

---

## 3. What was deliberately NOT changed

| Area | Status |
|---|---|
| `src/guardbound/attacks/*` (Crescendo, OppositeDay, Acronym, ActorAttack, RedQueen, adaptive, runner, prompts, rubric eval) | untouched |
| `src/guardbound/defense/*` (`SteeredLLMChat`, refusal detector, post-processing) | untouched |
| `src/guardbound/models/*` (dynamics, predictor, NBF, compat) | untouched |
| `src/guardbound/evaluation/*`, `experiments/*`, `analysis/*`, `data/*` | untouched |
| `configs/reproduction*.yaml`, `configs/default.yaml`, `configs/attacker_*.yaml` | untouched |
| `results/`, Phase 18 / Phase 18 Extension / Phase 19 scripts and raw outputs | untouched |
| NBF checkpoint (`cea1a75b…a136fe`), embedding model, threshold, eta, state init | untouched and re-verified |
| Judge-text artefacts (`configs/judge_prompt.txt`, `configs/judge_prompts/`) | untouched |

> **Correction (later audit).** An earlier version of this table listed
> "C1 prompt (`configs/judge_prompt.txt` / `configs/judge_prompts/`) untouched".
> That was a mislabel: C1 is **not** a judge prompt. It is the Phase-17 attacker
> generation-point instruction (mode `I1`), later adopted as an opt-in attacker
> condition in `src/guardbound/llm/attacker_arms.py`. See
> `PAPER_PROMPT_FIDELITY_AUDIT.md` §7.

Mechanically verified: `git status` shows only the files listed in §2 as
modified/untracked.

---

## 4. Architectural result

```
                     Guardbound (attacks, evaluator, NBF — unchanged)
                                      |
                                ChatLLM contract
                                generate(messages, temperature, json_format)
                                      |
                            provider_factory.build_role_llm
                          (one decision point, per role)
                         /                              \
                  local provider                   cloud provider
                       |                                 |
              HF / GLM4V / Qwen3.8 / Ornith15       OpenAI SDK
                (managed by ModelManager)        (Chat Completions)
                       |                                 |
              local weights                     gpt-4o / gpt-4o-mini
```

Each role resolves independently, so these all work with the same attack code:

```yaml
# mixed: cloud attacker + local target + cloud evaluator
models:
  attacker:  {provider: openai, model: gpt-4o-mini}
  target:    {provider: local,  model: meta-llama/Meta-Llama-3-8B-Instruct}
  evaluator: {provider: openai, model: gpt-4o-mini}
```

---

## 5. Cost and budget behaviour

* Cost per call = `uncached_input × input_rate + cached_input × cached_rate +
  output × output_rate`, from the YAML rate card.
* Cached tokens are clamped to the reported input count, so an inconsistent
  usage payload cannot overstate cost.
* Ledger records `by_role` (attacker / target / evaluator), `by_model`,
  `by_outcome`, `by_error_category`, plus totals and remaining budget.
* Every record carries model id, provider, endpoint, token counts, cached
  tokens, duration, request id, estimated cost, cost basis, temperature and
  `json_format`.
* Budget is enforced **locally, before** the request; no API-side limit is
  assumed. Once spent ≥ budget, the next call raises `BudgetExceededError`
  without touching the network (asserted in
  `test_budget_exhausted_blocks_the_request_before_it_is_sent`).

---

## 6. Reproducibility artefacts recorded per run

Written by `scripts/cloud_pilot.py` next to the results JSONL:

* model id per role, provider per role, API endpoint (`/v1/chat/completions`)
* structured-output mode, per-call temperatures, `max_turns`, attacks, success score
* dataset path, number of goals, sampling rule
* NBF checkpoint path + **sha256** + dims + threshold + eta + `steer_target` + mode
* config path + config sha256, rate-card version/as_of/verified
* protocol block (`retries: 0`, `repair: NONE`, `fallback: NONE`, …)
* environment (python, platform, package versions), git commit
* `result_kind` (`research run` vs `HARNESS VALIDATION ONLY`)
* per-conversation: turns, filtered flags, rubric scores, refusal count, NBF
  candidate scores/verdicts, evaluator outcome telemetry
* usage summary JSON + `UsageRecord` JSONL
