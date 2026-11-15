# CLOUD_SETUP_GUIDE

> **PARTLY SUPERSEDED BY PHASE 22.** `--budget-usd` was removed together with the
> custom `$1.50` cap (the paper specifies no API budget; usage is reported, never
> enforced). Where this guide shows `--budget-usd N`, omit the flag. Evaluator
> temperatures of 0.0 describe the legacy default; the paper-reproduction config
> uses 0.7. See `PHASE_22_IMPLEMENTATION_REPORT.md`.

Step-by-step from a clean checkout to a controlled cloud experiment.

Everything before step 5 costs nothing. Steps 5–8 spend money and each one is
gated behind an explicit flag.

---

## 0. Prerequisites

```bash
pip install -r requirements.txt
pip install -e .                     # or rely on tests/conftest.py's src/ shim
python -m pytest tests/test_cloud_provider.py tests/test_cloud_usage.py \
                   tests/test_cloud_research_integration.py -q
# expected: 120 passed, no network, no API key
```

The cloud provider needs `openai` (official SDK) plus `httpx`, which it depends
on. The NBF needs `torch` and `sentence-transformers`. Cloud-only runs with
NBF off need neither.

> **Windows/conda note.** If Python aborts with
> `OMP: Error #15: Initializing libiomp5md.dll`, two OpenMP runtimes are loaded.
> Set `KMP_DUPLICATE_LIB_OK=TRUE` for the session (see `.env.example`) — but
> treat it as a workaround, not a fix.

---

## 1. Provide the API key

The key is read from an environment variable at request time. It is never read
from a file in this repository, never written to a log, never echoed in an error,
and every emitted message passes through credential redaction.

**bash / zsh**

```bash
export OPENAI_API_KEY='...'      # from your shell, not from a repo file
```

**PowerShell**

```powershell
$env:OPENAI_API_KEY = '...'
```

Do **not**:

* commit a key, or put one in `configs/*.yaml`;
* pass one as a command-line argument (it lands in shell history);
* paste one into a chat, an issue, or a log.

`.env` is git-ignored (`.env.example` is tracked as a template with an empty
value). If you use a `.env` loader, load it into your shell environment — the
scripts read only `os.environ`.

Verify presence without revealing anything:

```bash
python -c "import os;print('present (len=%d)' % len(os.environ['OPENAI_API_KEY']) if os.environ.get('OPENAI_API_KEY') else 'absent')"
```

---

## 2. Verify the price card (do this before spending)

Open `configs/cloud_pricing.yaml`.

1. Check each model's `input_per_1m`, `cached_input_per_1m`, `output_per_1m`
   against the provider's official pricing page (the `source:` field).
2. Confirm the model ids you intend to use still exist. `gpt-4o` and
   `gpt-4o-mini` are legacy ids — prefer the dated snapshots
   (`gpt-4o-2024-08-06`, `gpt-4o-mini-2024-07-18`) for reproducibility.
3. Set `verified: true` and update `as_of`.

Until you do, every cost figure is labelled `unverified_rate_card` and the
scripts print a warning.

---

## 3. Dry run — plan everything, send nothing

```bash
python scripts/cloud_smoke_test.py --dry-run
python scripts/cloud_pilot.py --dry-run --variant cloud_baseline_a_nbf_off
python scripts/cloud_pilot.py --dry-run --variant cloud_baseline_b_nbf_on
python scripts/cloud_model_qualification.py --dry-run
```

The pilot dry run prints the resolved roles, temperatures, round budget, attack
list, goal count, NBF identity, rate-card status and budget warnings. For the
NBF-on variant it also verifies the checkpoint sha256 and aborts on a mismatch.

---

## 4. Offline mock — prove the harness works, spend nothing

```bash
python scripts/cloud_pilot.py --offline-mock --variant cloud_baseline_a_nbf_off \
    --attack crescendo_paper --limit 1 --out-dir /tmp/mock_pilot
```

This runs the real attack classes and the real runner against deterministic fake
transports. Output is tagged
`result_kind: "HARNESS VALIDATION ONLY (offline mock — not a research result)"`.
It validates plumbing, not model behaviour.

The NBF-on path additionally requires a working `sentence-transformers` install,
because the pilot warms up the embedding model **before** the first paid request.

---

## 5. Paid smoke test — two tiny calls

```bash
python scripts/cloud_smoke_test.py \
    --confirm-paid-request \
    --model gpt-4o-mini \
    --budget-usd 0.10 \
    --out results/cloud/smoke.json
```

Checks, in order: key present (not printed) → one minimal text request → one
structured-JSON request → token usage → duration → request id → cost estimate →
model id → clean exit. Exits non-zero on any failure. Expect well under a cent.

If it fails, stop here. The failure is classified (`authentication`,
`rate_limit`, `context_length`, …) and the cost report says `unavailable` rather
than a fake zero.

---

## 6. Model qualification — before any formal experiment

```bash
python scripts/cloud_model_qualification.py \
    --confirm-paid-request \
    --model gpt-4o-mini \
    --budget-usd 1.00 \
    --out-dir results/qualification
```

Read `results/qualification/gpt-4o-mini.qualification.json`. A model is only
"qualified" when, at minimum:

* attacker-contract validity rate is high (a low rate means most runs will fail
  with `AttackGenerationError`, not produce data);
* generated questions are non-empty and question-shaped;
* the target answers without truncation;
* the evaluator is deterministic at temperature 0.0 for repeated identical input;
* the multi-turn fact is recalled;
* the NBF loads and scores deterministically.

Do not proceed on a model that fails these. Do **not** tune the configuration in
response to preliminary ASR — that turns a test into a search.

---

## 7. Pilot — 5 goals, one attack, both NBF variants

```bash
# Baseline A: NBF off
python scripts/cloud_pilot.py \
    --variant cloud_baseline_a_nbf_off \
    --attack crescendo_paper --limit 5 \
    --budget-usd 2.00 --confirm-paid-request

# Baseline B: NBF on (identical except NBF)
python scripts/cloud_pilot.py \
    --variant cloud_baseline_b_nbf_on \
    --attack crescendo_paper --limit 5 \
    --budget-usd 2.00 --confirm-paid-request
```

Outputs land in `results/cloud/gpt4o_mini/`:

```
cloud_baseline_a_nbf_off.jsonl            one record per conversation
cloud_baseline_a_nbf_off.manifest.json    full provenance
cloud_baseline_a_nbf_off.usage.json       cost/token summary by role
cloud_baseline_a_nbf_off.usage.jsonl      one row per API call
```

The pilot refuses to append to an existing results file, so a re-run cannot
silently mix runs. Use `--out-dir` for a new run.

---

## 8. Formal experiment

Only after steps 5–7 pass. Same command, no `--limit` (all 200 HarmBench goals),
all three attacks, an explicit budget:

```bash
python scripts/cloud_pilot.py \
    --config configs/cloud_research_gpt4o_mini.yaml \
    --variant cloud_baseline_a_nbf_off \
    --budget-usd 25.00 --confirm-paid-request

python scripts/cloud_pilot.py \
    --config configs/cloud_research_gpt4o_mini.yaml \
    --variant cloud_baseline_b_nbf_on \
    --budget-usd 25.00 --confirm-paid-request
```

200 goals × 3 attacks × 2 variants × up to 8 rounds ≈ 9,600 conversations at the
cap. Estimate the spend from the pilot's measured cost per conversation **before**
launching, and set the budget accordingly.

With the paper's models instead:

```bash
python scripts/cloud_pilot.py --config configs/cloud_research_reference.yaml ...
```

---

## 9. Interpreting results responsibly

* Configuration B results are a **model substitution**. Never report them as the
  paper's GPT-4o numbers.
* NBF-off vs NBF-on is the only thing that differs between the two variants —
  the pilot asserts this before spending.
* Contract failures (`json_parse_error`, `missing_score`, …) are separate from
  attack outcomes. Check `evaluator_telemetry` and `by_outcome` before computing
  ASR; a high contract-failure rate means the ASR is unreliable, not low.
* Cost figures carry `cost_basis`. `unverified_rate_card` means the numbers are
  estimates from an unconfirmed price table.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not set` | key not exported in this shell | `export OPENAI_API_KEY=...` |
| `cloud provider selected but environment variable ... is not set` | role config points at cloud, key absent | export the key or switch the role to `local` |
| `provider=cloud cannot honour local-only keys [...]` | local sampling keys left in a cloud role | remove the keys, or use `provider: local` |
| `Rate card not found` | wrong `--rate-card` path | fix the path or omit the flag |
| `no rate in the rate card` | model missing from the rate card while a budget is set | add the rate |
| `Experiment budget exhausted` | spend cap reached | raise the cap deliberately, or stop |
| `sha256 mismatch` | different NBF checkpoint than the validated one | restore `models_best_nbf_released.pth` |
| `ModuleNotFoundError: ... sentence-transformers` | embedding stack not installed or broken | install/repair; NBF-off runs are unaffected |
| `operator torchvision::nms does not exist` | torchvision ABI mismatch with torch | reinstall a matching torchvision |
| `OMP: Error #15` | duplicate OpenMP runtime (conda) | `KMP_DUPLICATE_LIB_OK=TRUE` |

---

## 11. What never happens

* No request is sent at import time, during tests, or on `--dry-run`.
* No retry, repair, fallback, regeneration or temperature coercion in the formal
  path (`retries = 0`, `max_retries = 0` on the SDK client).
* No key is printed, logged, committed, or written into a manifest.
* No historical experiment artefact is read, modified, or overwritten.
