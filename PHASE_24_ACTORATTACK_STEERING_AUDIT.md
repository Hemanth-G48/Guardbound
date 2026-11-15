# PHASE 24 — ACTORATTACK STEERING AUDIT (D-02)

**Status: FIXED and verified on both branches (steering enabled / disabled).**

The composite runner now takes `steer_target`, honours it, installs the official
attack-level candidate filter, and keeps the verified composite control flow.

---

## 1. Root cause

`run_actor_attack_composite` (added in Phase 22) had **no `steer_target`
parameter**, gated the steered wrapper on `barrier is not None and embed_fn is not
None` alone, and never called `attack.set_safety_filter(...)`. Consequences:

* R1's `nbf.steer_target: false` was ignored: ActorAttack was target-steered while
  Crescendo and Opposite-day were candidate-filtered **in the same run**.
* The official candidate filter was unreachable: `call_multi`'s guard
  (`if use_safety_filter and self._embed_fn is not None and self._barrier is not
  None`) was always false because `_embed_fn` was never installed — so even
  `use_safety_filter=True` was a **silent no-op**.

Measured (Phase 23): `TypeError: … unexpected keyword argument 'steer_target'`;
after a barrier-enabled run, `attack._embed_fn is None`,
`attack._use_safety_filter is False`, `attack._safety_threshold == 0.0`, while a
`SteeredLLMChat` had been constructed with `eta=0.001`.

---

## 2. Change (`src/guardbound/attacks/runner.py`, `run_actor_attack_composite`)

New signature (Phase 24 additions marked):

```python
def run_actor_attack_composite(
    attack, goal, target_llm, target_system="You are a helpful assistant.",
    embed_fn=None, barrier=None,
    eta: float = 0.0, max_turns: int = 8, temperature=DEFAULT_TEMPERATURE,
    target_llm_name="", attack_method="actor_attack",
    steer_target: bool = True,                 # NEW: honoured, not inferred
    use_safety_filter: bool | None = None,     # NEW: None -> "a barrier was given"
    safety_threshold_override: float | None = None,   # NEW: raw threshold (η-negated)
    allow_regeneration=False, regeneration_max=3,
) -> ActorAttackRun:
```

Wiring, mirroring the generic runner exactly:

```python
has_barrier = barrier is not None and embed_fn is not None
if use_safety_filter is None:
    use_safety_filter = has_barrier
if use_safety_filter and not has_barrier:
    raise ValueError("use_safety_filter=True requires both a barrier and an embed_fn")
use_steered = bool(use_safety_filter and steer_target)

safety_threshold = threshold_from_eta(eta)          # paper rule: admit iff h < -eta
if safety_threshold_override is not None:
    safety_threshold = float(safety_threshold_override)

if has_barrier and hasattr(attack, "set_safety_filter"):
    attack.set_safety_filter(embed_fn, barrier, safety_threshold)   # official filter installed
attack.nbf_candidate_scores = []
attack.nbf_candidate_verdicts = []

if use_steered:
    … SteeredLLMChat(..., eta=eta, max_turns=max_turns * actor_num) …
    target = _SteeredAttackTarget(steered, embed_fn, temperature)
```

`call_multi` / `summary` now receive `use_safety_filter=use_safety_filter` and
`safety_threshold=safety_threshold` (was `safety_threshold=eta`, un-negated).

`scripts/cloud_pilot.py` passes the resolved flags:

```python
steer_target=bool(nbf_cfg.get("steer_target", False)),   # false for R1
safety_threshold_override=raw_threshold,                 # -0.001 for R1
```

The generic runner (`run_attack`, `run_attack_with_backtracking`, `…_async`) is
unchanged apart from the shared threshold conversion; `git diff` shows no other
edits inside those functions.

---

## 3. Why this is the official mechanism, not a wrapper bolted on

The author's `attack_single` → `call_multi` with `args.safety_filtering` filters
**candidate queries inside the attack** and calls the target plainly. That is what
`use_safety_filter=True` + `set_safety_filter` now does: `call_multi` scores the
candidate with `self._calculate_score` and skips it on `unsafe_cutoff`;
`summary` does the same for summary queries. The steered wrapper is a *separate*
mechanism that is now used only when `steer_target` asks for it.

The composite control flow was not touched: `prepare_attack` once, one fresh
`[system]` conversation per actor, query chains with `N_retry = 3`, summary per
non-skipped actor, score-5 early stop, per-actor records, one behaviour per goal.

---

## 4. Executed verification

Deterministic doubles: a barrier with separately controllable scores for the
candidate filter (`predictor`) and the steered Q-filter (`h`), plus instrumented
counters for `SteeredLLMChat` construction/`chat`.

| Scenario | `steer_target` | barrier | Observed | Verdict |
|---|---|---|---|---|
| R1 path (steering off) | `False` | candidate h = −0.2 (admit) | wrapper constructed **0**, `_use_safety_filter=True`, `_embed_fn` set, `_safety_threshold=-0.001`, candidate score recorded, target called | PASS |
| R1 path, unsafe query | `False` | candidate h = +0.5 | wrapper **0**, target **not** called, `nbf_candidate_verdicts == [False]`, candidate scores non-empty | PASS |
| Steering requested | `True` | candidate h = −0.2 | wrapper constructed **1**, `eta=0.001` passed, `chat` called, target called (chain + summary = 2) | PASS |
| Steered filter blocks | `True` | candidate −0.2 (admit) / steered +0.5 (filter) | wrapper **1**, `chat` called, target **not** called, candidate verdict `[True]` | PASS |
| No barrier | `False` | — | wrapper **0**, `_use_safety_filter=False`, `_embed_fn=None`, target called | PASS |
| Filtering without a barrier | `True` | — | `ValueError: use_safety_filter=True requires both a barrier and an embed_fn` | PASS — no silent no-op |
| Control flow with steering on | `True` | −0.2 | `prepare_attack` once, 3 actor records `completed`, 3 summaries | PASS |
| Early stop with steering on | `True` | −0.2 | 1 actor record, `goal_achieved=True` | PASS |

Plus: `test_pilot_passes_the_r1_steering_flags` asserts the pilot resolves
`steer_target is False`, `uses_composite_execution is True`, and the raw threshold
`-0.001` for the R1 YAML.

---

## 5. Scope discipline

* No change to `ActorAttack.call_multi` / `summary` semantics beyond the shared
  `admits(...)` inequality and the `admitted` telemetry boolean (which carries the
  same meaning as before: `True` = accepted).
* No new attack, no change to the paper's attack set.
* Opposite-day and Acronym keep their own in-attack filters; they now use the same
  helper with the same raw threshold.
* `run_full_attack` (the official-composite transcription) is untouched and still
  matches the composite's call sequence — the Phase 22 contract test passes.

---

## 6. Regression coverage

In `tests/test_phase24_p0_fixes.py` (all passing):

```
test_composite_honours_steer_target_false
test_composite_candidate_filter_blocks_a_steer_disabled_query
test_composite_steers_when_requested
test_composite_steered_filter_blocks_the_target
test_composite_without_a_barrier_filters_nothing
test_composite_rejects_filtering_without_a_barrier
test_composite_control_flow_survives_steering
test_composite_early_stop_still_works_when_steered
test_pilot_passes_the_r1_steering_flags
```

Phase 22/23 tests that documented the old defect were updated to the corrected
behaviour (`tests/test_phase22_paper_fidelity.py::
test_composite_requires_both_barrier_and_embed_fn_to_steer` now supplies a real
barrier and asserts the wrapper is constructed with `eta` and the per-actor turn
guard — the assertion it always intended).

---

## 7. Frozen-artifact evidence

`git status --porcelain` is empty for `results/`, `runs/`, `checkpoints/`, `data/`
and `configs/reproduction*.yaml`. No checkpoint, checkpoint hash, architecture,
embedding model, initial state or score formula was touched.
