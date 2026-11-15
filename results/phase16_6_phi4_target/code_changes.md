# Phase 16.6 — code changes

```text
NONE APPLIED.
```

The phase stopped at the feasibility gate (section 4 of the phase brief):
`microsoft/phi-4` requires **29.319 GB** at the reference precision (bfloat16) and the
GPU provides **24.385 GB** free. Per the brief — *"If the model does not fit in the
available VRAM under the required precision, document
MODEL_NOT_FEASIBLE_AT_REQUIRED_PRECISION and stop. Do not introduce quantization
merely to force it to fit."* — no production file was modified.

## The change that was prepared but deliberately NOT applied

```yaml
# configs/reproduction_phase14_frozen.yaml  (and any successor config)
models:
  target:
    model: microsoft/Phi-4-mini-instruct     # <- would become microsoft/phi-4
```

Applying it would point the pipeline at a model that cannot be loaded, turning a
documented infeasibility into a runtime failure. It was not applied.

## What was NOT touched (verified)

| item | status |
| --- | --- |
| `configs/reproduction_phase14_frozen.yaml` | unchanged — SHA256 `27B660A7052054521A94E3430013D9863964E825157056D3861228250F19A358` |
| attack algorithms / prompts | unchanged |
| attacker model (Qwen3-4B) / evaluator model (Llama-3.2-3B) | unchanged |
| NBF architecture, checkpoint, threshold, filtering | unchanged — checkpoint SHA256 `CEA1A75BCEF4FC51…` |
| JSON parser / retry logic / F2 classification | unchanged (as fixed in Phase 16.5) |
| F3 explicit-sampling behaviour | unchanged (as fixed in Phase 16.5) |
| Track A results | unchanged — `FCB952F1598B697B…` (180), `C12A562A9454D83B…` (108) |
| HarmBench dataset / goal ordering | unchanged — SHA256 `AC789DE8859E755C…` |
| GPU residency architecture and gates | unchanged |

## Findings that may require a change later (NOT applied)

1. **G1 — the reference local target ceiling is 256** (`generate.py:37`). The frozen
   config's `max_new_tokens: 256` is reference-faithful. Phase 16.5's F1 change
   (`max_new_tokens: null`, context-budget bounded) departs from that reference
   value. Reverting the F1 change, or keeping it, is an operator decision.
2. **G2 — the reference local target uses `top_p=0.9`** (`generate.py:41`). Neither
   the frozen config (inherited ≈1.0) nor the Phase 16.5 config (explicit 1.0)
   matches it. A one-line declaration change would close the gap, but it changes
   sampling for every role and is therefore deferred to the operator.

Both are recorded in `generation_config_validation.json`. Neither is applied.
