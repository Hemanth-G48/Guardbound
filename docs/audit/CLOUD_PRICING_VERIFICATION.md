# CLOUD_PRICING_VERIFICATION

Verification of `configs/cloud_pricing.yaml` against current official OpenAI
documentation. **Verified 2026-11-10.** `verified: true` is set because every
model Guardbound can run has a price read from an authoritative current source.

---

## 1. Sources (both read on 2026-11-10)

| # | Source | URL | Used for |
|---|---|---|---|
| 1 | API pricing table | `https://platform.openai.com/docs/pricing` → `https://developers.openai.com/api/docs/pricing` (`…/pricing.md`) | per-1M-token input / cached-input / output prices |
| 2 | Model pages | `https://developers.openai.com/api/docs/models/gpt-4o`, `…/gpt-4o-mini` | cross-check of prices; default snapshot; **deprecation status** |
| 3 | Deprecations page | `https://developers.openai.com/api/docs/deprecations` | announced shutdown dates; already-retired models |

`https://openai.com/api/pricing/` was also requested. It now redirects to the
**business/ChatGPT** pricing page (seat pricing), which contains no per-token API
rates, so it is not a usable source for this purpose. The figures below come from
the two documentation sources above, which agree with each other.

---

## 2. Verified rates

Units: **USD per 1,000,000 tokens**. Currency: **USD**. Basis: Standard tier
(the tier this repository's default configuration implicitly uses — no Batch,
Flex, Fast or Ultrafast modifiers are applied).

| Model id | Input | Cached input | Output | Basis |
|---|---:|---:|---:|---|
| `gpt-4o` | 2.50 | 1.25 | 10.00 | read directly (source 1 and 2) |
| `gpt-4o-2024-08-06` | 2.50 | 1.25 | 10.00 | inherited from the `gpt-4o` family rate |
| `gpt-4o-mini` | 0.15 | 0.075 | 0.60 | read directly (source 1 and 2) |
| `gpt-4o-mini-2024-07-18` | 0.15 | 0.075 | 0.60 | inherited from the `gpt-4o-mini` family rate |

Quoted from source 1 (Standard pricing table):

```text
| gpt-4o      | $2.50 | $10.00 | $1.25 |
| gpt-4o-mini | $0.15 | $0.60  | $0.075 |
```

and from the per-model pages:

```text
gpt-4o        Input $2.5   Cached input $1.25   Output $10     (1M tokens)
gpt-4o-mini   Input $0.15  Cached input $0.075  Output $0.6    (1M tokens)
```

---

## 3. Pricing assumptions, stated explicitly

1. **Snapshot pricing is inherited, not independently published.** The official
   pages price the *aliases* (`gpt-4o`, `gpt-4o-mini`) and do not list dated
   snapshots separately. Guardbound therefore resolves a snapshot id to its
   family rate by longest-prefix match. This is an **assumption** (that a
   snapshot is billed at its family rate), not a directly published figure; the
   rate card marks those entries `derived_from: <family>`.
2. **Standard tier only.** The table above is the Standard price. Batch (50 %
   cheaper), Flex/Fast/Ultrafast variants exist and are *not* modelled. A run
   using them would be mis-costed in the conservative direction for Batch and
   optimistically for Fast.
3. **Cached input is a real category, not a discount guess.** Both sources
   publish a cached-input rate, and the API reports
   `usage.prompt_tokens_details.cached_tokens`, which the client records. When
   cached tokens are reported, they are billed at the cached rate and deducted
   from the full-price input count.
4. **No cached rate is invented where none is published.** The rate card carries
   an explicit `null` when the provider publishes no cached-input price (e.g. it
   is `-` for several reasoning models). A `null` cached rate means cached tokens
   are billed at the normal input rate — the conservative choice, never zero.
5. **Unknown prices stay unknown.** A model with no entry gets
   `estimated_cost_usd = null` and `cost_basis = "unavailable"`. It is never
   recorded as `0`, because a silently-zero cost makes the budget guard useless.
6. **No API-side spending limit is assumed.** Enforcement is local, before each
   request.

---

## 4. Models deliberately absent from the rate card

| Model | Reason |
|---|---|
| `gpt-3.5-turbo-0125` | **Shut down 2026-10-23** (deprecations page). No price exists; a stale number would silently mis-cost a run. |
| `o1-2024-12-17` | **Shut down 2026-10-23**. Same reasoning. |
| `gpt-6-luna` | The officially recommended replacement for both GPT-4o models. **Not** configured or priced here: adopting it would be a second, separate model substitution that needs its own verification and its own substitution label. |

If an experiment is ever pointed at a model without a rate, the run fails loudly
rather than guessing.

---

## 5. Deprecation — a pricing-adjacent constraint that matters more

Both models Guardbound uses are deprecated with a published shutdown date:

```text
2026-10-14 announcement:
  gpt-4o      (gpt-4o-2024-08-06, gpt-4o-2024-11-20)  shut down 2027-01-14
  gpt-4o-mini (gpt-4o-mini-2024-07-18)                shut down 2027-01-14
  recommended replacement: gpt-6-luna
```

Consequences, recorded because they bound reproducibility and not just cost:

* the experiments can run **today** — a deprecated model still serves requests
  until its shutdown date;
* the reproducibility window closes **2027-01-14**, so every manifest records an
  execution timestamp and `scripts/cloud_pilot.py` prints the shutdown warning
  before a paid run (`_cloud_common.pricing_warnings`);
* the date is carried on `ModelRate.shutdown_date` and surfaced by
  `test_shutdown_model_produces_a_warning_before_a_paid_run`.

---

## 6. Cost calculation

```
cost = (input_tokens - cached_tokens) * input_rate
     +  cached_tokens                  * cached_rate        # or input_rate if null
     +  output_tokens                  * output_rate        # per 1M, divided by 1e6
```

* `cached_tokens` is clamped to `input_tokens`, so an inconsistent usage payload
  cannot overstate cost.
* The estimate is rounded to 8 decimal places.
* Every `UsageRecord` carries `estimated_cost_usd` **and** `cost_basis`
  (`rate_card` now that the card is verified, `unverified_rate_card` otherwise,
  `unavailable` when the model has no rate).

### Deterministic worked example (asserted in tests)

`tests/test_cloud_pricing_and_snapshots.py`:

```
input_tokens = 1000, cached = 400, output = 500

gpt-4o-mini-2024-07-18:
  600 * 0.15/1e6 + 400 * 0.075/1e6 + 500 * 0.60/1e6  = 0.00042 USD

gpt-4o-2024-08-06:
  600 * 2.50/1e6 + 400 * 1.25/1e6  + 500 * 10.00/1e6 = 0.00700 USD
```

---

## 7. What the API actually exposes for this calculation

| Need | Field | Present |
|---|---|---|
| input tokens | `usage.prompt_tokens` | yes |
| cached input tokens | `usage.prompt_tokens_details.cached_tokens` | yes (0 when no cache hit) |
| output tokens | `usage.completion_tokens` | yes |
| total | `usage.total_tokens` | yes |
| reasoning tokens | `usage.completion_tokens_details.reasoning_tokens` | yes (recorded, not billed separately here) |
| request id | `response.id` | yes |
| resolved model revision | — | **no** — recorded as a limitation |

All of the above are captured in `OpenAIChatLLM.last_request_stats` and in every
`UsageRecord`, so a cost figure can always be traced back to the tokens that
produced it.

---

## 8. Verdict

| Item | Status |
|---|---|
| `verified: true` justified | **yes** — every configured model's price read from an official current source |
| Snapshot pricing | inherited from family rate (**documented assumption**) |
| Unpriced models | reported as `null`, never `0` |
| Budget guard uses the verified rates | **yes** — same `RateCard` drives estimation and `check_budget` |
| Cost arithmetic tested deterministically | **yes** — 3 arithmetic tests + ledger tests |
| Model lifetime recorded | **yes** — shutdown date on the rate and in every manifest |
