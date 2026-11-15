# Phase 16.1 — Ornith-1.5-9B Structured-Output / Multi-Turn Diagnostic

**Date:** 2026-09-28
**Type:** diagnostic qualification study. **No ASR is computed, no attack-success
experiment is run, and the 90-run pilot was NOT launched.**

Four controlled configurations, identical in every other respect. The only
independent variables are `enable_thinking` and the structured-output mode.

```text
A  enable_thinking = false  + structured_output = constrained_json
B  enable_thinking = false  + structured_output = unconstrained
C  enable_thinking = true   + structured_output = constrained_json
D  enable_thinking = true   + structured_output = unconstrained
```

Held identical across all four: model revision, dtype, quantization, temperature
(0.7), `max_new_tokens` (256), system/attacker prompts, message construction,
conversation history, goals, JSON schema, parser, stopping behaviour, GPU runtime,
and the sequential residency strategy.

---

## 1. Reproducibility (per configuration)

| | A | B | C | D |
|---|---|---|---|---|
| Model | `ornith-ai/Ornith-1.5-9B` | same | same | same |
| Revision | `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` | same | same | same |
| dtype / quantization | bfloat16 / none | same | same | same |
| Backend | `Ornith15ChatLLM` | same | same | same |
| `chat_template_kwargs` | `{enable_thinking: false}` | `{enable_thinking: false}` | `{enable_thinking: true}` | `{enable_thinking: true}` |
| `structured_output_mode` | `constrained_json` | `None` | `constrained_json` | `None` |
| temperature / max_new_tokens | 0.7 / 256 | same | same | same |
| Residency | sequential | sequential | sequential | sequential |
| Seeds | `derive_seed(goal_id, "crescendo", "off")` + turn index — **identical across configurations** (paired) | same | same | same |
| Stage A goals | 0–19, file order | same | same | same |
| Stage B goals | 0–7, file order | same | same | same |
| Hardware | RTX 4500 Ada, 24.57 GB | same | same | same |

Target feedback for Stage B is **replayed** so every configuration sees identical
input: the recorded Track A NBF-OFF Crescendo target responses (read-only) where a
recording exists, and a fixed placeholder string for goal 1 (whose Track A run was
a `JSON_PARSE_ERROR` and recorded no turns). Per-goal provenance is in each
`configuration.json`.

---

## 2. Stage A — 20 isolated generations per configuration

| Stage A metric | A | B | C | D |
|---|---:|---:|---:|---:|
| Generations | 20 | 20 | 20 | 20 |
| **Valid JSON** | **15** | **17** | **7** | **0** |
| JSON validity rate | **0.75** | **0.85** | **0.35** | **0.00** |
| Malformed JSON (parser-rejected) | 5 | 3 | 13 | 18 |
| Missing required field | 0 | 0 | 0 | 0 |
| Wrong field type | 0 | 0 | 0 | 0 |
| Empty output | 0 | 0 | 0 | 0 |
| Reasoning-only output | 0 | 0 | 0 | 2 |
| Thinking marker present in reply | 1 | 0 | 0 | 2 |
| Extra text around JSON | 0 | 3 | 0 | 0 |
| Mean latency (s) | 7.18 | 5.66 | 7.03 | 16.82 |

## 3. Failure shapes (observation only — nothing was repaired)

Every non-parsing reply, both stages, classified by shape:

| Shape of the non-parsing reply | A | B | C | D |
|---|---:|---:|---:|---:|
| Truncated — no closing brace (budget exhausted) | 3 | 2 | 6 | **28** |
| Complete but trailing comma before `}` | 2 | 5 | 10 | 0 |
| Complete but key space-prefixed (`"  generatedQuestion"`) | **5** | 0 | 0 | 0 |
| Complete but otherwise invalid (incl. stray quotes) | 1 | 0 | 3 | 0 |
| Empty string | 0 | 0 | 1 | 0 |
| **Total non-parsing replies** (Stage A + Stage B) | 11 | 7 | 20 | 28 |

The shape labels are a heuristic applied to the preserved raw text (recorded in
`summary.json` under `analysis.failure_shapes`); the authoritative artifacts are
the raw files themselves.

Representative raw evidence (verbatim, unmodified) is preserved for every
generation in `config_*/stage_a_raw.jsonl` and `config_*/stage_b_raw.jsonl`.

## 4. Stage B — 8 multi-turn sequences per configuration

| Stage B metric | A | B | C | D |
|---|---:|---:|---:|---:|
| Sequences | 8 | 8 | 8 | 8 |
| Turns attempted | 14 | 19 | 16 | 8 |
| Turns successfully generated | 7 | 15 | 9 | 0 |
| **Turn completion rate** | 0.50 | **0.79** | 0.56 | **0.00** |
| Context failures | 7 | 4 | 7 | 8 |
| Context failure rate | 0.50 | 0.21 | 0.44 | 1.00 |
| Sequences completed (3/3 turns) | 2 | 4 | 3 | 0 |
| Sequence completion rate | 0.25 | **0.50** | 0.38 | **0.00** |
| — malformed syntax | 6 | 4 | 4 | 8 |
| — missing required field | 1 | 0 | 0 | 0 |
| — empty | 0 | 0 | 1 | 0 |
| — reasoning-only | 0 | 0 | 2 | 0 |
| Goal preserved in system message | True | True | True | True |
| Role ordering correct | True | True | True | True |
| Empty queries | 0 | 0 | 0 | 0 |
| Repeated queries | 0 | 0 | 0 | 0 |
| Mean latency per turn (s) | 8.34 | 7.53 | 8.82 | 16.66 |

Message structure was verified from the messages actually handed to the backend —
turn 1 `[system, user]`, turn 2 `[system, user, assistant, user]`, turn 3
`[system, user, assistant, user, assistant, user]` — with the previous attacker
query as the assistant turn and the replayed target reply inside the existing
"The last response was: …" user message. The attack layer built these messages;
no alternative history format was constructed.

## 5. GPU / residency (per configuration)

| | A | B | C | D |
|---|---:|---:|---:|---:|
| Load time (s) | 16.03 | 15.94 | 15.91 | 15.93 |
| Unload time (s) | 0.22 | 0.21 | 0.20 | 0.21 |
| Peak VRAM (GB) | 19.964 | 19.963 | 19.983 | 19.507 |
| Max resident models | 1 | 1 | 1 | 1 |
| Load refusals | 0 | 0 | 0 | 0 |
| OOM | 0 | 0 | 0 | 0 |
| CUDA errors | 0 | 0 | 0 | 0 |
| Residents after unload | 0 | 0 | 0 | 0 |

Sequential residency held in every configuration; every VRAM guard remained
active (nothing weakened, no quantization, no offloading).

---

## 6. Comparison table (§19)

| Metric | A: Think OFF + Constrained | B: Think OFF + Unconstrained | C: Think ON + Constrained | D: Think ON + Unconstrained |
|---|---:|---:|---:|---:|
| Isolated generations | 20 | 20 | 20 | 20 |
| Valid JSON | 15 | 17 | 7 | 0 |
| JSON validity | 0.75 | 0.85 | 0.35 | 0.00 |
| Empty | 0 | 0 | 0 | 0 |
| Reasoning-only | 0 | 0 | 0 | 2 |
| Missing fields | 0 | 0 | 0 | 0 |
| Multi-turn sequences | 8 | 8 | 8 | 8 |
| Turns attempted | 14 | 19 | 16 | 8 |
| Context failures | 7 | 4 | 7 | 8 |
| Sequence completion | 0.25 | 0.50 | 0.38 | 0.00 |
| Goal preserved | True | True | True | True |
| Repeated queries | 0 | 0 | 0 | 0 |
| Mean latency (s) | 7.18 / 8.34 | 5.66 / 7.53 | 7.03 / 8.82 | 16.82 / 16.66 |
| Peak VRAM (GB) | 19.964 | 19.963 | 19.983 | 19.507 |

*(latency shown as Stage A / Stage B)*

No configuration is ranked or called best; the numbers are reported as measured.

---

## 7. Interpretation (§20)

**Question 1 — Does disabling thinking improve structured JSON validity?**
In this sample, yes, in three of the four paired comparisons. With
`constrained_json`: 15/20 (0.75) with thinking off versus 7/20 (0.35) with
thinking on. Unconstrained: 17/20 (0.85) versus 0/20 (0.00). Multi-turn turn
completion: constrained 0.50 vs 0.56 (essentially level), unconstrained 0.79 vs
0.00. The exception (Stage B, constrained) is within two turns of the total; the
largest and most consistent difference is in the unconstrained mode.

**Question 2 — Does constrained JSON improve or worsen Ornith's structured output?**
The observed pattern is an **interaction**, not a single-direction effect. With
thinking off, the unconstrained condition produced *more* valid isolated replies
(17/20 vs 15/20) and a higher turn-completion rate (0.79 vs 0.50). With thinking
on, it is the reverse: constrained 7/20 versus unconstrained 0/20, and 0.56 vs
0.00. Constrained decoding removed prose-wrapped replies entirely
(`extra_text` 0 in A/C vs 3 in B) but did not prevent the syntax failures.

**Question 3 — Does either setting improve multi-turn reliability?**
The highest turn-completion rate observed was B (15/19 generated; 4/8 sequences
completed 3/3) and the lowest was D (0/8; no turn generated). Crucially, the
multi-turn *structure* never caused a failure: goal preservation and role
ordering were correct in 100 % of generated turns, and there were zero empty or
repeated queries in every configuration. The multi-turn failures are the same
JSON syntax rejections seen in isolation, not context loss.

**Question 4 — What are the failures primarily?**
**Malformed syntax**, in every configuration. Across both stages: D 28/28
truncated (the model spends the 256-token budget on analysis prose and never
closes the object); C 20 rejected (10 trailing comma, 6 truncated, 3 otherwise
invalid, 1 empty); B 7 rejected (5 trailing comma, 2 truncated); A 11 rejected
(5 space-prefixed key, 3 truncated, 2 trailing comma, 1 otherwise invalid).
Missing required field occurred once (A, Stage B); wrong field type never; empty
once; reasoning-only output four times (2 in D's isolated probes, 2 in C's
multi-turn turns).

**Question 5 — Does the failure pattern persist across the same 8 goals?**
Partly. In the paired isolated set only **1 of 20 goals** produced the same
outcome under all four configurations — i.e. the configuration, not the goal,
determined the outcome for 19/20 goals. In the multi-turn set the failing goals
differ per configuration (A {0,1,2,4,5,6,7}, B {0,1,4,6}, C {0,2,3,4,5,6,7},
D {0–7}), but **goal 6 failed in all four configurations in both stages**, and
goal 0 failed in all four multi-turn runs. So there is no fixed "hard goal" set,
but a small number of goals fail regardless of configuration.

**Question 6 — Where does the problem primarily sit?**
Within the four conditions tested: the **model's own structured-output
behaviour** is the limiting factor, with **thinking on amplifying it**. Thinking
on is associated with the largest degradation (unconstrained validity 0.85 →
0.00), while constrained decoding does not rescue it and its association changes
sign depending on the thinking setting. Multi-turn operation is not the source —
the failures are JSON syntax rejections, and context handling was correct
throughout. Even in the most favourable configuration tested, 3 of 20 isolated
replies and 4 of 19 turns were rejected, and the frozen parser (unchanged) turned
every one of them into a `JSON_PARSE_ERROR` with the raw output preserved.

*No causal claim is made beyond what this four-condition, n=20 / n=8-per-condition
experiment supports.*

---

## 8. Status

```text
80 isolated generations completed
32 multi-turn sequences completed
4 controlled configurations tested
8 multi-turn sequences per configuration
Track A unchanged
No parser repair
No retries
No fallback
No prompt modification
No ASR study launched
90-run pilot NOT LAUNCHED
```

Track A integrity, verified before and after: `results/phase14/batch00.jsonl` and
`results/phase15_pilot_30/trackA_frozen/batch00_goals000-029_180records.jsonl`
both `FCB952F1598B697B033D605B0791CFBB…`; the pre-resume snapshot
`C12A562A9454D83B6CC0EE93…`; frozen config `27B660A7052054521A94E3430013D986…`
with `config_hash = 61ea203c20e68c51`. Attack, NBF, target and evaluator code
mtimes all predate this phase.

## 9. Evidence locations

```text
results/phase16_1_ornith_diagnostic/
    config_A_thinking_off_constrained/  configuration.json, stage_a.json,
                                        stage_b.json, stage_a_raw.jsonl,
                                        stage_b_raw.jsonl
    config_B_thinking_off_unconstrained/   (same layout)
    config_C_thinking_on_constrained/      (same layout)
    config_D_thinking_on_unconstrained/    (same layout)
    summary.json          machine-readable cross-configuration summary
    diagnostic_report.md  this file
```

Nothing was written to `results/phase14/`, `results/phase15_pilot_30/` or
`results/phase15_glm46v/`, and the earlier Ornith qualification evidence was not
overwritten.

```text
PHASE 16.1 COMPLETE — 4-CONDITION ORNITH DIAGNOSTIC COMPLETE —
8 MULTI-TURN SEQUENCES PER CONDITION — 90-RUN PILOT NOT LAUNCHED
```
