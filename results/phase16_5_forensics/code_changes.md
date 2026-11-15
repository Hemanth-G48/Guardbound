# Phase 16.5 — code changes (F1, F2, F3)

Only these files were changed. No attack algorithm, prompt, NBF, threshold, dataset,
success criterion, seed policy, denominator rule or failure semantic was altered.

## F3 — explicit sampling parameters (FIXED)

| file | change |
| --- | --- |
| `src/guardbound/llm/local_client.py` | `HFLocalChatLLM.__init__` accepts `top_p`, `top_k`, `do_sample`. Declared values are passed to `generate()`; undeclared ones are not passed but are *reported* as `inherited_from_generation_config`. New `generation_parameters()` reports what the backend actually uses. |
| `src/guardbound/llm/provider_factory.py` | forwards `top_p` / `top_k` / `do_sample` from the role config. |
| `scripts/phase14_full_reproduction.py` | metadata no longer hardcodes `top_p: 1.0`; it reports each role's declared value or `inherited_from_generation_config`, plus `output_cap_declared` and `do_sample` per role. |

Before: the loader never passed `top_p`, so Qwen3-4B sampled with its own
`top_p=0.8/top_k=20` and Llama-3.2-3B with its own `top_p=0.9` while the metadata
claimed `1.0`. After: configs that declare sampling get exactly those values, and
every run's metadata states which values were inherited instead of assuming.

## F1 — target output cap (BLOCKED)

| file | change |
| --- | --- |
| `src/guardbound/llm/local_client.py` | `max_new_tokens=None` now means "no artificial cap": the call site passes the remaining **context budget** (`min(config.max_position_embeddings, tokenizer.model_max_length) - input_len`), which is the same class of bound the reference API has. An integer still means exactly that cap (frozen behaviour unchanged). |
| `configs/reproduction_phase16_5_forensics.yaml` | declares the target with `max_new_tokens: null`. |

No replacement fixed ceiling was introduced. The measured result is that this
target does not reach EOS on these prompts (see the report, section A), so F1 is
reported **BLOCKED** and the configuration carries an explicit do-not-run warning.

## F2 — failure classification (FIXED)

| file | change |
| --- | --- |
| `src/guardbound/attacks/opposite_day.py` | `generate_opposite_day_step` raises `AttackGenerationError` when the attacker reply is not parseable JSON, instead of returning `("", "")`. |
| `src/guardbound/attacks/acronym.py` | `generate_acronym_step`: same change. |

Unchanged on purpose: parsed dicts keep the lenient `.get(..., "")` semantics, so a
*valid* JSON reply with an intentionally empty query is still an ordinary empty
query; Crescendo keeps its stricter both-fields-required contract. No retry, no
repair, no regeneration, no prompt change, no budget change.

## Tests

`tests/test_phase16_5_forensics.py` (34 tests) covers F1/F2/F3 as required by §20.
One pre-existing parity test was updated because it depended on the corrected
behaviour (see the report, section G).
