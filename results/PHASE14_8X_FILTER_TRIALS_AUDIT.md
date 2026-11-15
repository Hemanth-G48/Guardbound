# Phase 14.8.x — Crescendo `filter_trials` Telemetry Audit (READ-ONLY)

**Date:** 2026-09-23
**Mode:** **READ-ONLY.** No code was modified. Stopping for approval, per Step 1.
**Scope:** explain why Crescendo records `filter_trials: null`, confirm the real filtering
configuration, and propose the smallest telemetry-only correction.

---

## 1. Required audit answers

```text
Actual Crescendo filtering trials      : 1 evaluation per generated candidate
                                         (declared config value = 3)
Actual OppositeDay filtering trials    : 1 evaluation per generated candidate
                                         (declared config value = 3)
Actual Acronym filtering trials        : 1 evaluation per generated candidate
                                         (declared config value = 3)

Telemetry lookup key for Crescendo     : attack_short == "crescendo"
                                         (ATTACK_KEYS["crescendo_paper"])
Actual configuration key for Crescendo : cfg["nbf"]["trials"]["crescendo"] == 3
                                         (same name — the config key is NOT the problem)

Root cause                             : TWO compounding defects, both in
                                         scripts/phase14_full_reproduction.py:
                                         (a) build_metadata drops Crescendo from the
                                             metadata dict entirely, because it filters
                                             cfg["nbf"]["trials"] by `k in ATTACK_KEYS`
                                             — a test of ATTACK_KEYS' *keys*
                                             ({crescendo_paper, opposite_day, acronym})
                                             against config keys that are attack-short
                                             names ({crescendo, opposite_day, acronym}).
                                             "crescendo" is not a key of ATTACK_KEYS, so
                                             it is filtered out before any lookup.
                                         (b) the record builder looks the value up by
                                             attack_short == "crescendo", which would miss
                                             even if (a) were fixed differently.

Proposed telemetry-only correction     : in build_metadata, filter by the config keys that
                                         correspond to real attacks:
                                             if k in ATTACK_KEYS.values()
                                         Metadata then becomes
                                             {crescendo: 3, opposite_day: 3, acronym: 3}
                                         and BOTH existing lookup sites
                                         (lines 670 and 952) resolve correctly with no
                                         change. One-token change; no hardcoded 3.
```

---

## 2. Where the configuration lives

`configs/reproduction_phase14_frozen.yaml`, `nbf.trials` (line 85):

```yaml
nbf:
  trials:
    crescendo: 3
    actor_attack: 10
    opposite_day: 3
    acronym: 3
```

Independent confirmation from the existing test suite —
`tests/test_reproduction_config.py:146-149` asserts exactly this, including
`trials.get("crescendo") == 3`.

**Note the config key is `crescendo`, not `crescendo_paper`.** The naming mismatch is *not*
config-vs-lookup; it is `ATTACK_KEYS`' keys (which are module names) vs its values (which are the
short attack names used everywhere else).

## 3. Where `filter_trials` is populated

| Site | Line | Code |
|---|---:|---|
| Metadata construction | **805** | `"filter_trials": {k: v for k, v in cfg["nbf"]["trials"].items() if k in ATTACK_KEYS}` |
| Completed-run record | **670** | `filter_trials=(metadata.get("filter_trials") or {}).get(attack_short)` |
| Failure record | **952** | `filter_trials=(metadata.get("filter_trials") or {}).get(attack_short)` |

`ATTACK_KEYS`:

```python
ATTACK_KEYS = {
    "crescendo_paper": "crescendo",     # key = module name, value = attack_short
    "opposite_day":    "opposite_day",
    "acronym":         "acronym",
}
```

### Why the lookup returns `None` — traced

```
cfg["nbf"]["trials"]          = {crescendo: 3, actor_attack: 10, opposite_day: 3, acronym: 3}
list(ATTACK_KEYS)             = [crescendo_paper, opposite_day, acronym]      # the KEYS

k = "crescendo"     -> "crescendo" in ATTACK_KEYS      -> False  -> DROPPED
k = "actor_attack"  -> "actor_attack" in ATTACK_KEYS   -> False  -> dropped (correct)
k = "opposite_day"  -> "opposite_day" in ATTACK_KEYS   -> True   -> kept
k = "acronym"       -> "acronym" in ATTACK_KEYS        -> True   -> kept

metadata["filter_trials"] = {opposite_day: 3, acronym: 3}      # crescendo is GONE

(metadata["filter_trials"]).get("crescendo")  ->  None         # the observed null
(metadata["filter_trials"]).get("opposite_day") -> 3
(metadata["filter_trials"]).get("acronym")      -> 3
```

This reproduces the Phase 14.7 baseline exactly: `filter_trials` was `3` in 12 records (opposite_day
×6 + acronym ×6) and `None` in 6 (crescendo ×6). **Defect (a) is the root cause**; fixing only the
lookup key would still yield `None`, because the value was already discarded.

## 4. Does the correction touch runtime filtering? — NO

Two independent reasons, both verified from source:

**(i) `cfg["nbf"]["trials"]` is read by no execution path.** A repo-wide search for `trials` shows it
in only three kinds of place:

| Use | Location |
|---|---|
| Config validation (`required_fields`) | `scripts/run_reproduction.py:718-721`, `tests/test_reproduction_config.py` |
| Record metadata | `scripts/phase14_full_reproduction.py:805` |
| Docstrings/comments | `opposite_day.py:424,439`, `acronym.py:469,483`, `phase14_full_reproduction.py:1151` |

The runner (`src/guardbound/attacks/runner.py`) contains **no reference to `trials` at all**. The
executed filter budget is a hardcoded `range(3)` in `runner.py:696` / `:989` and in the two attacks'
own `_filter_prompt_with_nbf` methods. Changing a metadata-derived value therefore **cannot** change
filtering behaviour.

**(ii) The change is confined to a dict comprehension inside `build_metadata`.** It touches no
attack, no runner, no NBF code, and no generation parameter.

## 5. What the audit found beyond the brief — worth your attention before I patch

### 5.1 The declared value `3` does not describe the executed loop

Both runner filter blocks are:

```python
for index_ in range(3):
    ssm_nbf_score = calculate_score_from_dialog(messages, query)
    ... record telemetry ...
    if ssm_nbf_score < safety_threshold:
        break                 # safe -> accept
    else:
        unsafe_cutoff = True
        break                 # unsafe -> filter
```

**Both branches break**, so the loop executes exactly **one** iteration in every case. The `range(3)`
never reaches a second pass — and there is no regeneration inside it; `query` is fixed before entry.
Regeneration happens in the enclosing `while round_number + _num_filtering < max_turns` loop (runner)
or `while attempts < self._max_turns` loop (`opposite_day.py:570`, `acronym.py:624`).

**This is already documented in the repository**, in `opposite_day.py:561-566`:

> *"Official candidate filtering: 3 trials, accept iff score < threshold. Mirrors
> `run_opposite_day`: the author's loop **evaluates the generated prompt once** (the `else` branch also
> breaks), and on `unsafe_cutoff` the prompt is filtered and the attack regenerates a new prompt."*

So the executed model is **1 evaluation per candidate, with regeneration**, for all three attacks —
not 3 trials per turn. This is consistent with the Phase 14.7 telemetry: `ON_crescendo_000` recorded
18 candidate scores over 7 accepted turns + 1 filter + 10 refusal retries = 18 loop iterations, one
evaluation each.

**Consequence for this patch:** `filter_trials = 3` will correctly report the **declared
configuration value** from the frozen config — which is what the field names alongside it
(`threshold`, `max_rounds`, `top_p`, `dtype`) also are. It will **not** describe the number of barrier
evaluations executed per candidate, which is 1. That runtime quantity is already reported separately
by `nbf_scores` / `filtered_queries` / `filter_count`.

I am flagging this because the phase states *"`filter_trials` ... must accurately describe the
configuration actually used by the experiment."* Under the reading **"the declared frozen
configuration"**, recording `3` is correct and I recommend proceeding. Under the reading **"the number
of trials the code executes"**, `3` would be inaccurate for all three attacks, and the correct value
would be `1`. **I will not choose between those readings unilaterally** — it determines what the
column means for the 1,200-run study.

Crucially, either way **the correction does not change experimental behaviour**, and **I am not
proposing to change the `range(3)` loop**. The repo already treats it as the official author
structure.

### 5.2 OppositeDay and Acronym already report `3` — but for a partially different reason

They report `3` because their config key happens to equal their `attack_short`, so they survive the
`k in ATTACK_KEYS` filter. They pass through the same buggy predicate. The proposed fix makes the
filter correct for all three uniformly rather than by coincidence.

## 6. Confirmations requested by Step 1

| Check | Result |
|---|---|
| 1. Where filtering-trial config is defined | `configs/reproduction_phase14_frozen.yaml` `nbf.trials` (line 85); inherited unchanged from `configs/reproduction_three_model.yaml` |
| 2. Exact config key/name used there | `crescendo` (also `opposite_day`, `acronym`, `actor_attack`) |
| 3. Where `filter_trials` is populated | `build_metadata:805`, consumed at `:670` (completed) and `:952` (failure) |
| 4. Why the lookup yields `None` | `build_metadata` discards `crescendo` before lookup (§3) |
| 5. Crescendo's actual filtering configuration | Declared `3`; executed `1` evaluation per candidate with regeneration (§5.1) |
| 6. OppositeDay / Acronym already report `3` | **Confirmed** — 12 records at `3` in the Phase 14.7 baseline |
| 7. Correction affects telemetry only | **Confirmed** — `nbf.trials` is read by no execution path (§4) |

## 7. Proposed change (for approval — not applied)

**File:** `scripts/phase14_full_reproduction.py`, `build_metadata`, line 805.

```diff
-        "filter_trials": {k: v for k, v in cfg["nbf"]["trials"].items() if k in ATTACK_KEYS},
+        # Config keys are attack-short names ({crescendo, opposite_day, acronym,
+        # actor_attack}); ATTACK_KEYS maps module name -> attack_short. Filtering
+        # by ATTACK_KEYS' KEYS therefore dropped "crescendo" (whose module name is
+        # "crescendo_paper") before the per-run lookup could read it. Compare
+        # against the values instead, so every real attack resolves.
+        "filter_trials": {k: v for k, v in cfg["nbf"]["trials"].items()
+                          if k in ATTACK_KEYS.values()},
```

**Effect:** `metadata["filter_trials"]` becomes `{crescendo: 3, opposite_day: 3, acronym: 3}`;
`actor_attack` stays excluded (correctly — it is not a Phase 14 attack). **Both existing lookup sites
resolve unchanged** — no second edit, no hardcoded `3`.

**Expected diff:** one predicate, one comment. Same file already modified in Phase 14.8. No `src/`
change. No test change required for existing behaviour.

**Proposed tests (Step 3), to be added only after approval:**

```
crescendo_paper -> filter_trials == 3   (the regression this phase exists for)
opposite_day    -> filter_trials == 3
acronym         -> filter_trials == 3
actor_attack    -> absent from filter_trials (not a Phase 14 attack)
build_metadata filter_trials is keyed by attack_short, not module name
```
plus a source-level guard that the predicate compares against `ATTACK_KEYS.values()`, so the
`crescendo` vs `crescendo_paper` mismatch cannot be silently reintroduced.

---

## STOP

Per Step 1: **stopped after the audit, awaiting approval.** No code was modified.

**Two things I need a decision on before I apply the patch:**

1. **Approve the one-line predicate fix** in `build_metadata`.
2. **Confirm the meaning of `filter_trials`**: the *declared config value* (`3`, my recommendation,
   consistent with its sibling fields) or the *executed evaluation count* (`1`). This changes what the
   column means for the whole study, and I would rather you choose than guess.

Independent of (2), the correction is telemetry-only: `nbf.trials` is read by no execution path, so
**runtime filtering cannot change**.

## Caveats stated honestly

1. **`GenuineIntel`-style claim avoided:** I did not run the filter loop to count iterations. The "1
   per candidate" reading is established from source (both branches `break`; the repo's own docstring
   says the author's loop evaluates once) and is corroborated arithmetically by the Phase 14.7
   telemetry (18 scores = 7 turns + 1 filter + 10 refusal retries). A direct count would need an
   instrumented run.
2. **`actor_attack` is excluded by design** (it is `LOCAL_MODEL_CAPABILITY_LIMIT` and not a Phase 14
   attack), so its `trials: 10` correctly does not appear in the metadata. Not a defect.
3. **Both lookup sites are in the same file** (completed and failure records), so one metadata fix
   covers both — this was verified by reading both, not assumed from the Phase 14.8 refactor.
4. **No claim is made about the official author code** beyond what the repository already documents;
   the `range(3)` structure is treated as intentional here and is explicitly out of scope.
