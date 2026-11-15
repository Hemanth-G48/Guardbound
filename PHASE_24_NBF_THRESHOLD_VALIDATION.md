# PHASE 24 — NBF THRESHOLD VALIDATION (D-01)

**Status: FIXED and independently verified.**

The paper's Q-filter rule is now defined in exactly one place and used by every
filter site. An executable boundary table proves the decision at and around
`h = −η` for both mechanisms.

---

## 1. Root cause

Phase 23 measured the defect with a barrier whose score was exactly `0.0`:

| Path | Code (before) | Decision with η = 1e-3 |
|---|---|---|
| `SteeredLLMChat.chat` | `filtered = (h + eta) >= 0` | `h = 0` → filtered ✅ paper-faithful |
| generic runner candidate filter | `safety_threshold = -eta` (correct default) **but** the pilot overrode it with `+nbf.threshold = +0.001` (`cloud_pilot.py:218`) | `0 < +0.001` → **admitted** ❌ |
| `ActorAttack.call_multi` / `summary` | composite passed `safety_threshold = +eta` (`runner.py:1288/1299`) | `0 < +0.001` → **admitted** ❌ |
| Opposite-day / Acronym `_filter_prompt_with_nbf` | `score < self._safety_threshold` with the same overridden value | same inversion ❌ |

Two distinct mistakes, one shared origin:

1. **The pilot treated η as a raw admission threshold.** The runners take
   `admit iff h < threshold`; the configuration's `nbf.threshold` is η (a
   non-negative margin). Passing `+η` where `−η` was required inverts the band
   `h ∈ [−η, +η)` — exactly the band η exists to control.
2. **The composite path never negated at all**, and the OD/Acronym in-attack
   filters inherited whichever value they were given. Nothing forced the sites to
   agree, so a future edit could re-introduce the split.

Because every frozen local experiment used η = 0, `+0` and `−0` coincide and no
historical result is affected.

---

## 2. Change

New module `src/guardbound/defense/filter_rule.py` (46 lines) owns both the
inequality and the conversion:

```python
def threshold_from_eta(eta: float) -> float:      # raw threshold: -eta
    return -float(eta)

def admits(score: float, threshold: float) -> bool:   # the shared inequality
    return float(score) < float(threshold)

def is_admitted(score, eta) -> bool: return admits(score, threshold_from_eta(eta))
def is_filtered(score, eta) -> bool: return not is_admitted(score, eta)
```

Every filter site now goes through it:

| File | Function | Change |
|---|---|---|
| `defense/steered_chat.py` | `SteeredLLMChat.chat` | `filtered = is_filtered(h_value, self.eta)` (identical maths, single source) |
| `attacks/runner.py` | `run_attack_with_backtracking`, `…_async` | `safety_threshold = threshold_from_eta(eta)` (was the inline `-eta if eta > 0 else 0.0`) |
| `attacks/runner.py` | candidate block (sync + async) | `admitted = admits(ssm_nbf_score, safety_threshold)`; telemetry records `admitted`; log margin is now `h + η` |
| `attacks/runner.py` | `run_actor_attack_composite` | derives `threshold_from_eta(eta)` from η instead of using `+eta`; honours a raw override |
| `attacks/actor_attack.py` | `call_multi`, `summary` | `admits(ssm_nbf_score, safety_threshold)`; telemetry records `admitted` |
| `attacks/opposite_day.py` | `_filter_prompt_with_nbf` | `admits(ssm_nbf_score, self._safety_threshold)` |
| `attacks/acronym.py` | `_filter_prompt_with_nbf` | same |
| `scripts/_cloud_common.py` | `raw_threshold_from_eta` (new) | the **only** place a configuration η becomes a raw threshold |
| `scripts/cloud_pilot.py` | `run_one` | converts `nbf.threshold` once (`raw_threshold`) and passes it to both generic and composite runners |

`scripts/cloud_pilot.py` conversion, verbatim:

```python
raw_threshold = (
    None if threshold is None else common.raw_threshold_from_eta(float(threshold))
)
```

`src/guardbound/paper_protocol.py` now also refuses to resolve an R1 configuration
whose two margin keys disagree, and exposes the derived value for the record:

```python
if nbf.get("threshold") is not None and float(nbf["threshold"]) != float(eta):
    raise PaperProtocolError("nbf.threshold and nbf.eta both denote … must be equal")
…
"raw_admission_threshold": -float(eta),
"filter_rule": "filtered iff h + eta >= 0 (admitted iff h < -eta)",
```

`eta < 0` is rejected outright.

Docstrings that stated the score orientation backwards (`runner.py`,
`actor_attack.py`: "lower = more unsafe") were corrected — that inversion is a
plausible origin of the defect.

---

## 3. Boundary table — executed

`η = 0.001`. Two mechanisms, same rule.

| `h` | Expected | `filter_rule.is_filtered` | `SteeredLLMChat.chat` | attack-level filter |
|---:|---|---|---|---|
| `0.0` | Filter | filtered | filtered, target not called | filtered, target not called |
| `-0.0005` | Filter | filtered | filtered | filtered |
| `-0.001` | Filter (strict) | filtered | filtered | filtered / ambiguous by 1e-17¹ |
| `-0.0015` | Admit | admitted | admitted, target called | admitted, target called |
| `-0.002` | Admit | admitted | admitted | admitted |
| `+0.001` | Filter | filtered | filtered | filtered |

¹ The attack-level test reaches the barrier through `softmax(logits)`, so the
reconstructed `h` at exactly `−η` can differ by ~1e-17. The test therefore asserts
the *decision follows the observed score* (`verdict == not is_filtered(score)`),
and the exact boundary is asserted on `filter_rule` and `SteeredLLMChat` where the
value is exact (float64). A dedicated case proves the exact-zero decision
end-to-end: uniform logits give `h = 0.0`, and the attack-level filter returns
`[False]` with zero target calls.

Strict inequality is pinned separately:
`is_filtered(-η, η) is True`, `is_filtered(-η - 1e-12, η) is False`.

---

## 4. Configuration propagation

| Check | Result |
|---|---|
| `nbf.eta == nbf.threshold == 0.001` | PASS |
| `resolve_paper_reproduction(cfg)["eta"] == 1e-3` | PASS |
| `["raw_admission_threshold"] == -1e-3` | PASS |
| `nbf_settings(cfg, "r1_paper")` → eta 0.001, threshold 0.001 | PASS |
| `raw_threshold_from_eta(0.001) == -0.001` | PASS |
| Runner without an override (`eta=0.001`) sets `_safety_threshold == -0.001` | PASS — no double negation |
| Runner with the pilot's override sets `_safety_threshold == -0.001` | PASS |
| Composite with `eta=0.001` sets `_safety_threshold == -0.001` | PASS |
| Config with `threshold != eta` | rejected (`PaperProtocolError`) |
| Config with `eta < 0` | rejected |

---

## 5. Historical behaviour preserved

* Every frozen configuration uses η = 0 → `threshold_from_eta(0) == -0.0`, and
  `h < -0.0` is exactly `h < 0.0`. No decision changes.
* `safety_threshold_override` keeps its Phase 19 meaning (a **raw** threshold), so
  `use_safety_filter`-style tests that pass `-0.5` or `±2.0` behave identically.
* The parity suite's sign-convention test asserts the same invariant through the
  new helper (`threshold_from_eta(0.5) == -0.5` plus the runner's source line).

---

## 6. Regression coverage

`tests/test_phase24_p0_fixes.py`:

* `test_filter_rule_boundary_table` — 6 rows × 3 predicates.
* `test_filter_rule_is_strict_at_the_boundary`.
* `test_filter_rule_matches_the_paper_expression` — 5 scores × 4 η values
  cross-checked against `(h + η) >= 0`.
* `test_steered_llm_chat_uses_the_same_rule` — 6 rows, target-call count included.
* `test_attack_level_filter_uses_the_same_rule` — 6 rows through the real generic
  runner and the real Crescendo attack.
* `test_attack_level_filter_is_exact_at_zero`.
* `test_runner_converts_eta_when_no_override_is_given`.
* `test_r1_configuration_propagates_eta_without_inversion`.
* `test_r1_rejects_a_threshold_that_differs_from_eta`,
  `test_r1_rejects_a_negative_margin`.

All pass (64/64 in the new file; 1131 collected overall).

---

## 7. Frozen-artifact evidence

```
git status --porcelain -- results runs checkpoints data \
    configs/reproduction_phase12.yaml configs/reproduction_three_model.yaml \
    configs/reproduction_phase16_5_forensics.yaml nbf_original_stuff
# -> empty
```

Checkpoint sha256 re-verified in Phase 23 and untouched:
`cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe`.
No architecture, embedding, initial-state or score-formula change.
