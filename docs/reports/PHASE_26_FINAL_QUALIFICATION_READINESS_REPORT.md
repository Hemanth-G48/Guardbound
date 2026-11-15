# PHASE 26 — FINAL QUALIFICATION READINESS REPORT

**Status: `READY FOR MINIMAL QUALIFICATION`.**

Every Phase 25 P1 gate defect is fixed and independently verified: the NBF readiness
checks are a mandatory decision, a skipped or failed NBF probe cannot yield
`QUALIFIED`, model/condition overrides are rejected on paper configurations, and the
verdict is derived from per-probe statuses including a global `probes_complete`. The
local NBF gate runs before any client is constructed, the attack-level candidate
filter is exercised with local metrics, and per-call usage is persisted
incrementally (and survives an interruption).

No paid request, qualification, smoke test, pilot or experiment was run. No API key
was read, requested or exposed. No budget cap exists or was introduced.

Audit date: 2026-11-10 · Windows 11 · Python 3.13.9 · branch `main` @
`89ed03f` (+ uncommitted Phases 20-26).

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Can a skipped NBF probe still yield `QUALIFIED`? | **No** — local gate `INSUFFICIENT_DATA`, `probes_complete = false`, overall `NOT_QUALIFIED` (verified) |
| Can a failed NBF load/check still yield `QUALIFIED`? | **No** — `local` decision `NOT_QUALIFIED`, run stops **before** any client (0 constructed, verified) |
| Can `--model` substitute GPT-4o-mini on an R1 config? | **No** — exit 2 before validation side effects and before any client |
| Can `--attacker-prompt-condition c1_generation_point` be used on R1? | **No** — same rejection |
| Does the verdict depend on an explicit probe-completeness check? | **Yes** — derived from six per-probe statuses (`attacker`, `actor_attack_pre_attack`, `target`, `evaluator`, `nbf_integration`, `candidate_filter`) |
| Do the local NBF checks run before paid probes? | **Yes** — checkpoint → embedder → score → margin → candidate filter, all before client construction |
| Is the attack-level candidate filter actually exercised? | **Yes** — real barrier + real embedder + scripted attack + counting target; filtered → 0 target calls, admitted → ≥1 |
| Is per-call usage persisted? | **Yes** — `<stem>.usage.jsonl`, incrementally after each probe group; 60 rows in a full mock run, 36 retained after an interruption |
| Offline suite | **1164 collected / 1160 passed / 4 failed / 0 errors / 0 skipped** — the same four environmental failures, no new ones |
| Classification | **`READY FOR MINIMAL QUALIFICATION`** |

The qualification script is the only production file changed (plus one config field
and one threshold block); the attacks, NBF, prompts, dataset and frozen artifacts are
untouched.

---

## 2. Phase 25 discrepancy resolution

| ID | Severity | Status | Where |
|---|---|---|---|
| P1-1 NBF readiness not gated | P1 | **Fixed** | `scripts/cloud_model_qualification.py`: `MANDATORY_PROBES`, `_local_metrics`, `DECISION_ROLES`, `final_decision` |
| P1-2 CLI overrides accepted on paper configs | P1 | **Fixed** | `parse_args`/`main` — rejection before validation, key read and client construction |
| P1-3 no per-probe status / completion gate | P1 | **Fixed** | `probe_status`, `probes_status`, `probes_complete`, `final_decision` |
| P2-1 NBF checks after paid probes | P2 | **Fixed** | `main` — local gate before `build_single_role_llm`; failure returns 1 with a report |
| P2-2 per-call usage not persisted | P2 | **Fixed** | `_flush_usage` after the local gate and each probe group, plus `_persist_report` |
| P2-3 candidate filter not probed | P2 | **Fixed** | `probe_candidate_filter` (real barrier/embedder, scripted attack, counting target) |
| P3-1 `nbf.enabled` missing in the qualification config | P3 | **Fixed** | `configs/qualification_r1_gpt4o.yaml` → `nbf.enabled: true` (+ declared `local` thresholds) |
| P3-2 ambiguous NBF result keys | P3 | **Fixed** | `verified` → `checkpoint_verified`, `loaded` → `load_ok` |
| P3-3 no error category in probe samples | P3 | **Fixed** | `_stats` adds `error_category` / `error_message` |
| P3-4 interrupted pilot manifest counts | P3 | **Open** (outside this phase's scope; `complete: false` still prevents misreading) |
| P2-4 `h = −η` float-noise boundary | P2 | **Documented** (inherent to a float barrier; ~1e-7 band vs η = 1e-3) |
| Phase 23 D-10 empty-chain goal | P2 | **Accepted** (faithful to `attack_single`) |
| Phase 23 D-16 upstream HarmBench byte check | P3 | **Open** (needs network; the file hash is pinned) |

No P0 and no P1 remains.

---

## 3. NBF mandatory gate (P1-1)

**Root cause.** `probe_nbf_integration` returned a dict that only `main` printed;
`assemble_metrics` built the three model-role dictionaries and `overall_decision` was
computed from those alone. `--skip-nbf` returned `{"skipped": true}` and a load
failure returned `{"verified": true, "loaded": false}` — neither influenced the
verdict, so a run with an unusable barrier could report `QUALIFIED`.

**Correction** (`scripts/cloud_model_qualification.py`):

* `LOCAL_ROLE = "local"`, `DECISION_ROLES = (*ROLES, LOCAL_ROLE)`; a new `local`
  threshold block declares eight boolean checks.
* `_local_metrics(nbf, cf)` maps the two local probes onto
  `nbf_checkpoint_verified`, `nbf_loaded`, `nbf_embedding_dim_ok`,
  `nbf_score_valid`, `nbf_threshold_applied`, `nbf_deterministic`,
  `candidate_filter_ok`, `candidate_filter_both_branches` — each `1.0` / `0.0`, or
  `None` when it could not be measured.
* `probe_nbf_integration` now reports each check explicitly and verifies:
  checkpoint hash (`verify_nbf_checkpoint`), offline load, embedding dimension 768,
  finite score with the paper formula (`h = P(5) − max(P(1..4))`), determinism across
  two evaluations, and the margin (`eta == threshold`, `raw == -eta`, and for a paper
  config `eta == 0.001`).
* `decide` treats the local role as all-or-nothing: a **measured** violation is
  `NOT_QUALIFIED`; an **unmeasurable** check is `INSUFFICIENT_DATA` (the three model
  roles keep their Phase 24 partial-measurement behaviour).
* `final_decision` requires the local role plus every mandatory probe to pass.

**Outcome mapping — verified** (`tests/test_phase26_qualification_gate.py`):

| Situation | `probe/decision` | Overall |
|---|---|---|
| All checks pass (stub bundle, h = −0.4) | PASS / QUALIFIED | can be QUALIFIED |
| `--skip-nbf` | INSUFFICIENT_DATA | **never QUALIFIED** |
| Checkpoint missing | FAIL / NOT_QUALIFIED | **never QUALIFIED** |
| Checkpoint hash mismatch | FAIL / NOT_QUALIFIED | **never QUALIFIED** |
| Embedder unavailable | FAIL (`error_category: embedder`) | **never QUALIFIED** |
| `threshold ≠ eta` (incl. inverted sign) | FAIL | **never QUALIFIED** |
| `eta ≠ 0.001` on a paper config | FAIL | **never QUALIFIED** |
| Non-deterministic score | FAIL | **never QUALIFIED** |
| Determinism unmeasurable (`None`) | INSUFFICIENT_DATA | **never QUALIFIED** |
| Filter branch not exercised (`None`) | INSUFFICIENT_DATA | **never QUALIFIED** |

**Limitation.** The gate proves the local checks, not the remote model. It cannot
detect an account without GPT-4o access; that remains a paid-run observation (§14).

---

## 4. CLI override protection (P1-2)

**Root cause.** `main` validated the paper protocol first and then applied
`--model` (all three roles) and `args.attacker_prompt_condition`; both therefore
bypassed the approved R1 decisions.

**Correction.** Immediately after loading the config, before validation, before the
key check and before any client:

```python
if paper:
    forbidden = [f for f, given in (("--model", args.model),
                                    ("--attacker-prompt-condition",
                                     args.attacker_prompt_condition)) if given]
    if forbidden:
        print(f"Refusing {', '.join(forbidden)}: … is a paper reproduction "
              "configuration (paper_reproduction: true). …", file=sys.stderr)
        return 2
```

**Verified** (independent probe, §10):

| Invocation on `configs/qualification_r1_gpt4o.yaml` | Exit | Clients constructed | stderr |
|---|---:|---:|---|
| `--model gpt-4o-mini-2024-07-18` | **2** | **0** | `Refusing --model: … paper reproduction configuration …` |
| `--attacker-prompt-condition c1_generation_point` | **2** | **0** | names the flag |
| both | **2** | **0** | names both |
| no override, `--dry-run` | 0 | 0 | — |

**Legacy behaviour preserved.** `configs/cloud_qualification.yaml` (non-paper) still
accepts `--model`; verified with `--dry-run` (exit 0 and the overridden id printed).
No rejection is silent: the message names every offending flag and the config.

**Limitation.** A non-paper config can still substitute a model — that is its purpose,
and the report records the ids actually used.

---

## 5. Global probe-completeness (P1-3)

`MANDATORY_PROBES` declares six probes with the source result keys and the metrics
each contributes:

| Probe | Role thresholds | Metrics |
|---|---|---|
| `attacker` | attacker | contract validity, coverage, empty-question rate, multi-turn usability, request errors |
| `actor_attack_pre_attack` | attacker | payload validity, empty-chain rate |
| `target` | target | non-empty, truncation, 4-turn recall, request errors |
| `evaluator` | evaluator | rubric validity, determinism, agreement, refusal, disclaimer, structural violations, request errors |
| `nbf_integration` | local | the six NBF checks |
| `candidate_filter` | local | filter consistency, branch coverage |

`probe_status(name, metrics, thresholds)` → `PASS` (all metrics present and in
bounds), `FAIL` (a measured violation), `INSUFFICIENT_DATA` (a metric missing),
`NOT_EXECUTED` (no source result at all). `probes_status` produces a status for every
mandatory probe, and `probes_complete` is `all(status == PASS)`.

`final_decision(decisions, statuses, config_intact, required=None)`:
`NOT_QUALIFIED` if any role fails, the configuration was not intact, or any required
probe has no status; `NOT_QUALIFIED` if any required probe is `FAIL`;
`INSUFFICIENT_DATA` if any required probe is not `PASS`; `QUALIFIED` only when every
role and every required probe passes. The verdict is computed from the probe results;
there is no caller-supplied completion flag.

**Verified** (`test_final_decision_requires_roles_probes_and_config`, plus the
independent runs): a missing probe entry → not QUALIFIED; `INSUFFICIENT` →
`INSUFFICIENT_DATA`; `FAIL` → `NOT_QUALIFIED`; `config_intact=False` →
`NOT_QUALIFIED`; all pass → `QUALIFIED`.

---

## 6. NBF-first execution sequence (P2-1)

`main` now runs, in this order:

1. `parse_args`
2. `load_config`
3. **P1-2 override rejection** (paper configs only)
4. `validate_paper_config` → `config_intact`
5. resolve samples, models, arm, output paths, thresholds
6. print the plan; `--dry-run` returns 0 listing the local probes first
7. paid-confirmation gate (returns 2 without `--confirm-paid-request`)
8. API-key presence check (or the offline-mock transport)
9. usage ledger
10. **`probe_nbf_integration`** — checkpoint hash, offline load, dimension, score,
    determinism, margin
11. **`probe_candidate_filter`** — attack-level filtering with the real barrier
12. `local` decision + local-gate verdict; a `NOT_QUALIFIED` local gate writes the
    report (`stopped_before_paid_probes: true`) and returns **before any client**
13. client construction (attacker/target/evaluator)
14. attacker probes (contract, multi-turn, consistency, refusal judge, ActorAttack
    pre-attack) → usage flush
15. target probes → usage flush
16. evaluator probes → usage flush
17. metrics → per-role decisions (incl. `local`) → per-probe statuses →
    `probes_complete` → `final_decision`
18. report + `qualification.jsonl` + usage flush

**Verified.** With a missing checkpoint: exit 1, `clients_constructed = 0`,
`stopped_before_paid_probes = true`, `usage.calls = 0`. With the real barrier: the
local gate passes (decision `QUALIFIED`) and only then are the three clients built.
The paid gate remains mandatory (exit 2 without the flag).

**Deviation from the brief's numbering:** the brief places "initialize usage
telemetry" (5) before the API-key check; the implementation checks the key first
(step 8) because it is local and fails faster. Both precede every paid probe.

---

## 7. Candidate-filter probe (P2-3)

`probe_candidate_filter(cfg, skip)` — fully local:

* real checkpoint + real `all-mpnet-base-v2` embedder (cached loader, one load shared
  with the NBF probe);
* a deterministic scripted `ActorAttack` (`_StepJudgeAttacker`, step judge always
  `"successful"`) driving the **production** `ActorAttack.call_multi` with
  `use_safety_filter=True` and `safety_threshold = threshold_from_eta(eta)`;
* a counting target stub (`_CountingTarget`) — no provider is contacted.

Two stages:

| Stage | Barrier | Scores | Purpose |
|---|---|---|---|
| real scores | real checkpoint | 3 benign queries × {η = 0.001, η = 0} | every decision must match `is_filtered(h, margin)` and the target-call count |
| exact boundary | `_FixedHCandidateBarrier` | h = 0.0 → filtered, h = −2η → admitted | strict-inequality placement through the same attack code path |

Metrics: `candidate_filter_ok` (all executed cases consistent, no errors) and
`candidate_filter_both_branches` (`True` only when both a filtered and an admitted
case were observed; `None` otherwise → `INSUFFICIENT_DATA`).

**Measured with the real checkpoint/embedder:**

```
real filtered/admitted = 0/6    boundary = 1/1    ok = True
```

All six real cases were admitted (the three benign queries score below −η at both
margins), and the exact-boundary stage supplied the filtered branch: **h = 0.0 with
η = 0.001 produced zero target calls**; h = −2η produced a target call. So the filter
demonstrably *blocks* generation as well as admitting it, and the requirement
"filtered candidate → zero target calls / admitted → target call" holds on the real
attack path.

**Limitation (P3).** With these three queries no real score landed in the filtered
region at η = 0.001; the filtered branch is proven by the exact-boundary stage rather
than by a real score. That is a property of the probe corpus, not of the filter.

---

## 8. Per-call usage persistence (P2-2)

`_flush_usage(ledger, path)` appends every not-yet-written `UsageRecord` to
`<stem>.usage.jsonl`, tracking an offset per path so each record is written exactly
once per process. It is called after the local gate, after each paid probe group, and
by `_persist_report`.

**Verified:**

| Check | Result |
|---|---|
| Rows in a full offline-mock run | **60**, equal to `usage.calls` |
| Roles present | `attacker`, `target`, `evaluator` |
| Row keys | `role`, `provider`, `model`, `endpoint`, `input_tokens`, `cached_tokens`, `output_tokens`, `total_tokens`, `duration_s`, `request_id`, `outcome`, `error_category`, `error_message`, `estimated_cost_usd`, `cost_basis`, `usage_available`, `temperature`, `json_format`, `finish_reason`, `refusal`, `timestamp` |
| Unknown cost | `estimated_cost_usd = None` (never 0); `cost_basis = unavailable` |
| Interruption (KeyboardInterrupt inside the target probe) | `usage.jsonl` exists with **36** rows; **no** report written |
| Budget enforcement | none (no `budget_usd`, no blocking) |
| Provider payloads | unchanged (the ledger only observes; Phase 24 test) |

---

## 9. Configuration changes

`configs/qualification_r1_gpt4o.yaml`:

* `nbf.enabled: true` (P3-1) — agrees with the actual behaviour (the harness requires
  a passing local NBF gate; the key documents it for readers and for any pilot-style
  merge).
* a `thresholds.local` block declaring the eight boolean checks (mirrors
  `DEFAULT_THRESHOLDS["local"]`).
* nothing else changed: models, temperatures, prompt variant, attacker condition,
  attacks, training-only set, η = 0.001 / raw −0.001, 8 turns, 200 goals, no budget.

`configs/cloud_qualification.yaml` (legacy): unchanged; it inherits the `local`
defaults. Its tests still pass, including the "attacker passes, target fails"
discrimination check on the mock transport.

---

## 10. Independent verification evidence

Standalone probe (not the test module), instruments `common.build_single_role_llm`
and calls the real `main`/gate functions:

| Scenario | Exit | Clients | Observed |
|---|---:|---:|---|
| paper config + `--model gpt-4o-mini-2024-07-18` | **2** | **0** | refusal names the flag and "paper reproduction configuration" |
| paper config + `--attacker-prompt-condition c1_generation_point` | **2** | **0** | refusal names the flag |
| paper config + both | **2** | **0** | refusal names both |
| legacy config + `--model`, `--dry-run` | 0 | 0 | override applied (id printed) |
| paper config, `--dry-run` | 0 | 0 | "R1 protocol validated", local probes listed first |
| `--offline-mock --skip-nbf` | 1 | 3 | overall `NOT_QUALIFIED`, `probes_complete=false`, nbf `INSUFFICIENT_DATA`, local `INSUFFICIENT_DATA`, 60 usage calls |
| paper config, checkpoint missing | 1 | **0** | overall `NOT_QUALIFIED`, `stopped_before_paid_probes=true`, nbf `FAIL`, 0 usage calls |
| paper config, full run with the real barrier | 1 | 3 | local decision `QUALIFIED`; probe statuses `attacker PASS`, `actor_attack_pre_attack PASS`, `nbf_integration PASS`, `candidate_filter PASS`, `target FAIL`, `evaluator FAIL`; overall `NOT_QUALIFIED` because the offline mock cannot recall context or classify refusals |
| interruption inside the target probe | KeyboardInterrupt | 3 | `usage.jsonl` 36 rows, no report |
| external request lines in any stderr | — | — | **0** |

The full-run row is the key negative control: the local gate and the attacker role
pass, and the verdict is still `NOT_QUALIFIED` because two mandatory probes fail — the
harness discriminates rather than rubber-stamping.

---

## 11. Offline test results

```powershell
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=line --junitxml=_phase26_run1.xml
```

| Collected | Passed | Failed | Errors | Skipped | Time |
|---:|---:|---:|---:|---:|---:|
| **1164** | **1160** | **4** | **0** | **0** | 105.3 s |

| Comparison with the Phase 25 baseline | Result |
|---|---|
| Collected | 1131 → **1164** (+33 new gate tests) |
| Passed | 1127 → **1160** |
| Failed node IDs | **identical set** (four environmental) |
| Newly introduced failures | **none** |
| Tests modified / skipped / deleted | none |

The four known failures, compared by exact node ID:

```
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured
tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid
tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present
```

Two require CUDA (torch is CPU-only); two require `Qwen/Qwen3.5-4B` weights that are
not on disk. They were not modified, skipped or deselected.

New coverage (`tests/test_phase26_qualification_gate.py`, 33 tests): NBF gate matrix
(skip, missing checkpoint, hash mismatch, embedder failure, threshold matrix incl. the
inverted sign, non-determinism, unmeasurable determinism/branch coverage), probe
status vocabulary and completeness, `final_decision` matrix, candidate-filter probe
(both branches, zero target calls when filtered), CLI rejection (three combinations +
legacy preservation, with clients forbidden), local-gate-before-client ordering,
usage JSONL shape and interruption retention, `nbf.enabled`, `error_category`.

The suite is network-free: model loads are stubbed in the gate tests, and the real
checkpoint + MPNet path remains covered by `tests/test_cloud_nbf_integration.py`
(7 tests). The independent probe in §10 ran with `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1` at process start and logged **zero** external request lines.

---

## 12. Frozen-artifact integrity

```
git status --porcelain -- results runs checkpoints data \
    configs/reproduction_phase12.yaml configs/reproduction_three_model.yaml \
    configs/reproduction_phase16_5_forensics.yaml nbf_original_stuff
# -> empty
```

| Artifact | State |
|---|---|
| `results/`, `runs/`, `checkpoints/`, `data/` | unchanged |
| Historical `reproduction*.yaml` | unchanged |
| Official checkpoint | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` (re-hashed) |
| Original author code (`nbf_original_stuff/`) | unchanged |
| Historical prompt variants | untouched (the `legacy` constants are read, never written) |
| Phase 22-25 reports | not edited (only `HANDOFF.md`'s status line was updated) |

No git write operation was performed.

---

## 13. Remaining discrepancies

| ID | Sev | Item | Observation | Action |
|---|---|---|---|---|
| P3-a | P3 | Real-score filtered branch not observed | the three probe queries score < −η at both margins, so the filtered branch is proven by the exact-boundary stage (h = 0 → 0 target calls) rather than by a real score | add a query whose real score lies in [−η, 0) to the probe corpus in a later phase, or accept as documented |
| P2-b | P2 | `h = −η` float noise | the decision at exactly −η follows the floating-point score (~1e-7 for a float32 barrier vs η = 1e-3); shared with the official implementation | none; documented |
| P3-c | P3 | Per-probe wire temperature not recorded | the probes pass the configured temperatures (source + tests) but the report records the configuration, not each call's payload | optional: include `temperature` in `_stats` |
| P3-d | P3 | Interrupted **pilot** leaves `goals_completed: 0` in the manifest | `complete: false` and `finished_at: null` still make the status unambiguous | Phase 27 housekeeping |
| — | — | Phase 23 D-10 (empty chain can score) | faithful to the official `attack_single`; accepted | analysis-side exclusion documented |
| — | — | Phase 23 D-16 (upstream HarmBench byte check) | needs network; the file hash is pinned | when network is authorised |

No P0, no P1, no gate defect.

---

## 14. Exact future qualification command

Validated by parsing, `--dry-run`, `--offline-mock` and a full local gate — **never
executed against a provider.**

```powershell
Set-Location "C:\Users\CSLAB\Desktop\hemanth\Guardbound"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"          # duplicate OpenMP runtime (conda)
$env:HF_HUB_OFFLINE = "1"                    # NBF embedder: cache only, no network
$env:TRANSFORMERS_OFFLINE = "1"
$env:OPENAI_API_KEY = "<paste in this shell only — never a file, never an argument>"

python scripts/cloud_model_qualification.py `
    --config configs/qualification_r1_gpt4o.yaml `
    --confirm-paid-request
```

| Property | Value |
|---|---|
| Working directory | `C:\Users\CSLAB\Desktop\hemanth\Guardbound` |
| Config | `configs/qualification_r1_gpt4o.yaml` (unchanged R1 stack) |
| Confirmation | `--confirm-paid-request` required; without it the script prints the threshold plan and exits **2** |
| Budget cap | none, and no such flag exists (`--budget-usd` is rejected) |
| Overrides | **do not pass** `--model`, `--attacker-prompt-condition` (exit 2) or `--skip-nbf` (forces `INSUFFICIENT_DATA`) |
| Outputs | `results/qualification_r1/<model>.baseline.qualification.json`, `…usage.jsonl`, `qualification.jsonl` |
| Expected volume | ≈60 requests `[N]`; cost on the order of a few US dollars at the verified GPT-4o rates — an estimate, **not a quote**, with no cap by design |
| Success | exit 0 and `overall: QUALIFIED` with `probes_complete: true` |
| Failure | exit 1 with `overall` `NOT_QUALIFIED`/`INSUFFICIENT_DATA`, the per-probe statuses, and (if the local gate failed) `stopped_before_paid_probes: true` and zero usage calls |

---

## 15. Final readiness classification

# `READY FOR MINIMAL QUALIFICATION`

| Criterion | Met? | Evidence |
|---|---|---|
| Phase 23 P0 fixes intact | **yes** | §11 suite; no change to the attack/NBF code paths in this phase |
| NBF readiness mandatory | **yes** | §3, §10 (skipped → `INSUFFICIENT_DATA`, failed → stop before clients) |
| All required probes must pass | **yes** | §5 `probes_complete` |
| Skipped/missing probe cannot yield `QUALIFIED` | **yes** | §10 rows 6-7; `test_final_decision_*` |
| Model / condition overrides blocked for R1 | **yes** | §4, §10 rows 1-3 (exit 2, 0 clients) |
| NBF checks before paid probes | **yes** | §6, §10 row 7 (0 clients on a local failure) |
| Candidate-filter behaviour tested | **yes** | §7 (real barrier + embedder; h = 0 → 0 target calls) |
| Per-call usage persistence | **yes** | §8 (60 rows; 36 retained on interruption) |
| No unresolved P0/P1 gate defect | **yes** | §13 |
| Offline suite vs baseline | **yes** | 1164 / 1160 / 4 / 0 / 0, same four node IDs |

**This classification is an implementation readiness decision only.** It does not
authorise an API request; a separate explicit decision is required (§16).

---

## 16. Explicit authorization requirement

Before the first paid qualification:

1. **Your explicit authorisation** for one qualification run of
   `configs/qualification_r1_gpt4o.yaml`. Nothing in this report grants it.
2. `OPENAI_API_KEY` exported in the shell only — never in a file, never as an
   argument, never logged; every message passes through credential redaction.
3. The exact command in §14, unmodified (no `--model`, no
   `--attacker-prompt-condition`, no `--skip-nbf`).
4. Accept that there is **no spending cap**: the cost is only knowable after the run,
   from the persisted usage summary and `usage.jsonl`.
5. Report the outcome back before any R1 experiment is considered: the report path,
   `overall`, `probes_complete`, the six per-probe statuses and the usage summary.
6. If the run returns `NOT_QUALIFIED`, the correct response is to report the failing
   metrics and choose a different model as an explicitly labelled substitution — not
   to relax a threshold, edit a prompt or tune the attack.

No credential was requested, read or exposed during this phase.
