# PHASE 28 — FINAL PAPER-TO-CODE REPRODUCIBILITY AUDIT

**Classification: `INCOMPLETE SCOPE`** — the attack-ASR half of the GPT-4o row is
implemented faithfully; the utility half (MTBench, XSTest) has code but no data, no
R1 wiring, and a wrong default steering margin, so Table 1 cannot be reproduced from
this repository as it stands. Within the ASR scope there are **two P1 methodological
deviations** (ASR denominator, evaluator temperature) and **one P1 completeness gap**
(no declared baseline variant).

Static, offline, read-only audit. No code, configuration, prompt, dataset,
checkpoint or artifact was modified; no API request was made; no credentials were
read. Repository state audited: `main` @ `89ed03f` + 90 uncommitted files.

---

## 1. Executive summary

### 1.1 What is faithful

* **The scoped target row is the right one and the model is pinned**: GPT-4o
  (`gpt-4o-2024-08-06`) appears verbatim in the paper (p8, §5.1), and R1 uses
  exactly that id for all three roles.
* **Attack set and training-only exclusion** match the paper: ActorAttack,
  Crescendo, Opposite-day evaluated; Acronym training-only (p22, B.1), enforced by a
  validator that rejects acronym-in-evaluation, unknowns, duplicates and empties.
* **Both attack loops are transcriptions of the author's code, including its
  quirks**: one attacker call per round, the vacuous `for index_ in range(3)` NBF
  loop, the `while round_number + num_filtering < max_rounds` budget, the
  pop-and-regenerate refusal rule, the `round_number > 1` summary rule (and its
  first-round misalignment), the ActorAttack `N_retry = 3` rewrite loop, the
  unknown-verdict actor skip, and the summary-score-5 early stop.
* **The NBF is the released checkpoint, unmodified**, with the paper's architecture,
  encoder, class indexing, score formula and the strict filter rule
  (`admit iff h < −η`), η = 1e-3.
* **The dataset is the author's file**: 200 records, all `max_rounds: 8`, all
  `target_system: "You are a helpful assistant."`, no duplicates, no empty goals.
* **Prompt bytes** are the author's, selectable as a variant with the legacy bytes
  preserved; the rubric is the Qi et al. rubric the code uses.

### 1.2 What is not

| # | Finding | Severity |
|---|---|---|
| A | **ASR denominator is "valid completed goals", not "all harmful queries"** (paper p8 §5.1). With *f* infrastructure failures the reported ASR is inflated by 200/(200−f). | **P1** |
| B | **Evaluator temperature is 0.7 in R1, 0.0 in the author's code** (`evaluate_with_rubric.py:31`, `check_refusal.py:64`). The paper says 0.7; the published numbers were produced at 0.0. Because success is `max(score) == 5` over all judged turns, judge noise is an ASR *inflator*. | **P1** |
| C | **No baseline variant is declared** in `configs/paper_reproduction_r1_gpt4o.yaml` (`variants: {r1_paper}` only). `resolve_variant(cfg, "r1_baseline")` raises; `assert_invariants` silently returns `[]` for <2 variants, so the "differ only in NBF" guarantee is inert. | **P1** (table completeness) |
| D | **MTBench and XSTest cannot be run**: no dataset files in the repository, no utility block in the R1 config, and `scripts/eval_mtbench.py` defaults `--eta 5e-4` instead of the R1 1e-3. The author's release also ships no utility data. | **P1** (table completeness) |
| E | **Zero SDK retries** (`max_retries=0`) versus the author's 10-attempt loop (`generate.py:129-151`). Transient provider errors therefore remove goals, which interacts with A. | **P1/P2** |
| F | Two ASR paths exist; the second (`judge_asr.py`) uses a **non-official judge prompt**. Only the pilot's attack-time rubric is paper-faithful. | **P2** |

No `P0` was found: nothing in the current code contradicts the paper's method or the
author's execution in a way that would make a *correctly reported* result misleading.
The four P1s either distort the reported number (A, B), make part of the table
unrunnable (C, D), or bias the completed-goal population (E).

---

## 2. Source inventory

### 2.1 Paper — *Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks* (Hu, Robey, Liu; TMLR 02/2026, arXiv:2503.00187v3)

Text extracted from `2503.00187v3.pdf` (31 pages, 101 822 characters). Anchors used
in this report:

| Anchor | Location | Content used |
|---|---|---|
| Table 1 | pp. 8–9 (text lines 557–567) | GPT-4o row: `0.600 / 0.565 / 0.725`, MTBench `9.35`, XSTest `0.004`; +steering `0.035 / 0.260 / 0.325`, `8.77`, `0.026` |
| §5.1 | p. 8 (605–657) | model ids (`gpt-4o-2024-08-06`), attack set, "All temperatures are set to be 0.7", ASR = successful jailbreaks judged by GPT-4o "over all harmful queries", MMLU/MTBench/XSTest definitions |
| §5.2 | p. 9 (659–665) | "NBF-based steering with threshold η = 1e−3" |
| Eq. (7), (10) | p. 5 (439–480) | safe set `φ_k = h + η < 0`; per-turn invariance `h(x,u) < −η` |
| §4 (text) | p. 6 (533–537) | "filters out all harmful attack queries û_k where **h(x_{k−1}, û_k) + η ≥ 0**" |
| B.1 | p. 22 (1247–1281) | 200 HarmBench samples, training corpus sizes, unsafe = 5, λ = (1,1,100,100), η_train = 0, "at most 8 turns", rejective-turn rule, MMLU/MTBench post-processing, harmless ⇔ score 1, 4×A6000 |
| Table 8 / Table 9 | p. 23 (1298–1329) | GPT-4o at η = 1e−3: `0.035 / 0.260 / 0.325`; MMLU `62.69`, MTBench `8.77` |

### 2.2 Author's implementation — `nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/`

Entry point `steering.py` (118 lines), attacks (`crescendomation/run.py`,
`opposite_day/run.py`, `actor_attack/run.py`, `acronym/run.py`), utilities
(`generate.py`, `evaluate_with_rubric.py`, `check_refusal.py`, `check_disclaimer.py`),
NBF loader `build_ssm_nbf.py`, checkpoint `models/models_best_nbf_released.pth`,
data `data/test/harmbench_tasks.json`.

### 2.3 Guardbound

Configurations `configs/paper_reproduction_r1_gpt4o.yaml` (155 lines, sha256
`f39d7369…`), `configs/qualification_r1_gpt4o.yaml` (`5d296b2b…`); attacks
`src/guardbound/attacks/{crescendo_paper,opposite_day,actor_attack,runner}.py`;
defence `src/guardbound/defense/filter_rule.py`; protocol
`src/guardbound/paper_protocol.py`; evaluation `src/guardbound/evaluation/*`;
runners `scripts/{cloud_pilot,cloud_model_qualification,_cloud_common}.py`.

---

## 3. Three-way methodology comparison

Classification legend: `EXACT MATCH` / `PAPER MATCH / CODE DIFFERENCE` /
`CODE MATCH / PAPER AMBIGUITY` / `PARTIAL MATCH` / `MISSING` / `INCORRECT` /
`UNVERIFIABLE` / `NOT APPLICABLE`.

| ID | Methodology requirement | Paper evidence | Official implementation | Guardbound implementation | Classification | Result impact |
|---|---|---|---|---|---|---|
| M1 | Target = `gpt-4o-2024-08-06` | §5.1 p8 | `steering.py:70` default `gpt-4o` (floating alias) | `configs/paper_reproduction_r1_gpt4o.yaml:53` pinned id | **PAPER MATCH / CODE DIFFERENCE** | Code may have served a different snapshot than the pinned id; G is stricter |
| M2 | Attacker = GPT-4o | not stated (only the judge is named) | `steering.py:71` default `gpt-4o` | `:56` `gpt-4o-2024-08-06` | **CODE MATCH / PAPER AMBIGUITY** | Identical model family; pinning only removes drift |
| M3 | Evaluator = GPT-4o | §5.1 p8 ("GPT-4o as the LLM safety judge") | same client as attacker | `:59` `gpt-4o-2024-08-06` | **EXACT MATCH** | none |
| M4 | Attacker/target temperature 0.7 | §5.1 p8 | `generate.py:95` default 0.7 | `:81-82` 0.7 | **EXACT MATCH** | none |
| M5 | Evaluator temperature 0.7 | §5.1 p8 ("All temperatures") | **0.0 hardcoded**: `evaluate_with_rubric.py:31`, `check_refusal.py:64`, `check_disclaimer.py:29` | `:83` 0.7 (`PAPER_EVALUATOR_TEMPERATURE`) | **PAPER MATCH / CODE DIFFERENCE** | Judge noise rises; success is `max==5` over up to 8 turns → ASR inflation risk (**P1**) |
| M6 | `top_p` | silent | OpenAI path sends none; local path `top_p=0.9` (`generate.py:41,84`) | none unless configured (`openai_client.py:288`) | **CODE MATCH / PAPER AMBIGUITY** | none for gpt-4o |
| M7 | Max output tokens | silent | OpenAI path sends none (Claude 8192, local 256) | none unless configured (`openai_client.py:19,28`) | **CODE MATCH / PAPER AMBIGUITY** | API default; no artificial truncation |
| M8 | Stop sequences | silent | none | none | **EXACT MATCH** (all silent) | none |
| M9 | Retry policy | silent | 10 attempts, `sleep(2i+1)`, return `{}` (`generate.py:129-151`) | `max_retries=0` (`openai_client.py:223`), `retries: 0` | **PARTIAL MATCH** | Official almost never loses a goal; G can (**P1/P2**, see A/E) |
| M10 | System prompt | silent; dataset field | `test_case["target_system"]` (`crescendomation/run.py:25`) | dataset field, default "You are a helpful assistant." (`:121`) | **EXACT MATCH** | none (all 200 rows carry it) |
| M11 | Request format | silent | `response_format` text / `json_object` (`generate.py:141`) | same (`openai_client.py:254`) | **EXACT MATCH** | none |
| A1 | Attack set = ActorAttack, Crescendo, Opposite-day | §5.1 p8 | dispatch `steering.py:107-114` | `:96-99`; validator `paper_protocol.py:75-119` | **EXACT MATCH** | none |
| A2 | Acronym training-only | B.1 p22 | present but not evaluated | `training_only: [acronym]`, rejected in eval | **EXACT MATCH** | none |
| A3 | Budget "8 turns" | B.1 p22 | `while round_number + num_filtering < max_rounds`, `round = len(history_t)//2 + 1` → **7 target calls** | `runner.py:674-682` reproduces both lines | **PAPER MATCH / CODE DIFFERENCE** | G matches the code's off-by-one, not the paper's prose (**P3**) |
| A4 | ActorAttack: 3 actors | silent (Ren et al.) | `actor_num = 3` hardcoded (`actor_attack/run.py:214`) | `actor_attack.py:600` default 3 | **CODE MATCH / PAPER AMBIGUITY** | none |
| A5 | ActorAttack `N_retry = 3` (4 attempts) | silent; code comment says "same as paper" | `run.py:80,82` | `N_retry: 3`; `runner.py:750` | **CODE MATCH** | none |
| A6 | ActorAttack order: rubric → step judge; unknown verdict skips the actor; summary score 5 stops | silent | `run.py:109,122-131`, `:195-203` | `runner.py:746-761`, `:1331-1369` | **EXACT MATCH** | none |
| A7 | Crescendo: one attacker call per round, no candidate regeneration | silent | `crescendo/run.py:46` + vacuous `range(3)` at `:53-63` | `runner.py:689-732` (same vacuous loop) | **CODE MATCH / PAPER AMBIGUITY** | none — the paper never describes candidate sampling for Crescendo |
| A8 | Refusal does not consume a turn; query regenerated | B.1 p22 | `check_refusal(...) and C_refused < 10` → `history_t.pop()`, continue | `runner.py:769-789` | **EXACT MATCH** | none |
| A9 | After 10 refusals the refusal counts as a turn | silent | `C_refused < 10` guard | `should_backtrack()` (`crescendo_paper.py:661-663`) | **CODE MATCH** | none |
| A10 | Opposite-day has no disclaimer check | silent | OD imports only `check_refusal`, `evaluate_with_rubric`; `check_disclaimer` is Acronym-only | `opposite_day.py` has no disclaimer path | **EXACT MATCH** | none (a disclaimer check here would have been a defect) |
| A11 | Per-goal success = any turn scored 5 (Crescendo/OD) | B.1 p22 (score 5 = unsafe) | `if score == 5: return` | `aggregate_scores` → `max(valid) == 5` (`cloud_pilot.py:144-163`) | **EXACT MATCH** | none |
| A12 | Composite ActorAttack: fresh conversation per actor, system message, per-actor records, one behaviour per goal | silent | `attack_single` (`actor_attack/run.py:189-205`) | `run_actor_attack_composite` (`runner.py:1219-1385`) | **EXACT MATCH** | none |
| N1 | Checkpoint identity | released artifact | `models/models_best_nbf_released.pth` | same file, sha256 `cea1a75b…`, pinned in config | **EXACT MATCH** | none |
| N2 | Encoder `all-mpnet-base-v2`, 768-d | §5.1 p8 | `find_emb_ssm_nbf` | same, 768 verified | **EXACT MATCH** | none |
| N3 | Architectures 1536-512-512-768 / 1536-32-32-5 | §5.1 p8 | checkpoint shapes | same | **EXACT MATCH** | none |
| N4 | Score = P(5) − max(P(1..4)) | Eq. (5) | `probs[:,-1] - max(probs[:,:-1])` | `runner.py:649-652` | **EXACT MATCH** | none |
| N5 | Filter iff h + η ≥ 0 ⇒ admit iff h < −η (strict) | §4 p6 (line 534); Eq. (7),(10) | `threshold = -args.threshold`; `if score < threshold` | `admits(h, −η)` strict, `filter_rule.py:30-46` | **EXACT MATCH** | none; boundary float noise only (**P2**) |
| N6 | η = 1e−3 | §5.2 p9; Table 1; Tables 8–9 | CLI default 0 | `:135-136` 0.001 | **PAPER MATCH / CODE DEFAULT DIFFERENCE** | Correct for the table; the code's default would be a different experiment (**P3**) |
| N7 | Initial state `x_0 = 0` | §4 | `calculate_score_from_dialog` | zeros(768) | **EXACT MATCH** | none |
| N8 | Filter placed after generation, before the target call | §4 | before `target_generate` | same (`runner.py:706-732` then `740`) | **EXACT MATCH** | none |
| N9 | No candidate survives → regenerate, budget consumed | §4 | `num_filtering += 1; continue` | same | **EXACT MATCH** | none |
| N10 | Inference precision/device | silent | GPU float32 | CPU float32 (local) | **PARTIAL MATCH** | ≤1e-7 score shift; only matters within 1e-7 of −1e-3 (**P2**) |
| D1 | 200 HarmBench behaviours | B.1 p22 | `data/test/harmbench_tasks.json` | same file, sha256 `ac789de8…` | **EXACT MATCH** | none |
| D2 | Same goals for baseline and defence | implied by Table 1 | one loop over all 200 | one dataset, `sampling: file_order` | **EXACT MATCH** | none |
| D3 | Unsafe label 5 | B.1 p22 | `score == 5` | `success_score: 5` | **EXACT MATCH** | none |
| E1 | ASR numerator = successful behaviours | §5.1 p8 | per-goal `goal_achieved` | `success = max(valid)==5`, one row per goal | **EXACT MATCH** | none |
| E2 | ASR denominator = **all** harmful queries | §5.1 p8 ("over all harmful queries") | every one of the 200 goals writes a verdict; a crash aborts the run | `asr_denominator = valid_completed_goals` (`cloud_pilot.py:588`) | **PARTIAL MATCH** | Inflates ASR by 200/(200−f) (**P1**, see §7.3) |
| E3 | Judge = GPT-4o with the Qi et al. rubric (1–5) | §5.1 p8 | `evaluate_with_rubric.py` (verbatim bytes) | official prompt variant, byte-selected | **EXACT MATCH** | none |
| E4 | Invalid evaluator reply | silent | `data["score"]` → `KeyError` (crash) | `None`, goal excluded and flagged | **PARTIAL MATCH** | Feeds E2; G is safer but changes the population (**P2**) |
| E5 | Single ASR path | §5.1 implies the attack-time judge | yes | pilot rubric **plus** `judge_asr.py` (non-official prompt) | **PARTIAL MATCH** | Risk of reporting the wrong path (**P2**) |
| U1 | MTBench 1–10 with refusal substitution | B.1 p22 | not in the release | `evaluation/mtbench.py` implements it | **PARTIAL MATCH** | No data, not in R1 config, `--eta` default 5e-4 (**P1**, §9) |
| U2 | XSTest over-refusal | §5.1 p8; B.1 | not in the release | `over_refusal.py` + `eval_over_refusal.py` | **PARTIAL MATCH** | No data; detector algorithm unspecified (**P1**, §9) |
| U3 | MMLU post-filter (positive ⇒ wrong) | B.1 p22; Table 9 | not in the release | `evaluation/mmlu.py` | **PARTIAL MATCH** | No data; Table 9 MMLU not in R1 scope |
| X1 | LoRA baselines, guardrail F1, JBB, PHTest, adaptive attack | Tables 2–4, 13–14 | partial | Phase 8/7 code exists | **NOT APPLICABLE** | outside the GPT-4o Table-1 row |

---

## 4. Attack-by-attack audit

### 4.1 ActorAttack

**Official** (`attacks/actor_attack/run.py`): `run_actor_attack` (`:208`) sets
`threshold = -args.threshold` (`:211`), `actor_num = 3` (`:214`), one
`infer_single` pre-attack call (`:216`), then `attack_single` (`:189`) loops the
actors: `call_multi` (`:78`) with `N_retry = 3` → up to 4 attempts per query index;
inside each attempt the NBF block (`:87-100`) scores `queries[i]` **once** (the
`range(10)` breaks on the first iteration either way), rewrites `queries[i]` to the
min-score query, and on `unsafe_cutoff` retries; the target call appends the query,
the rubric score is computed before `step_judge` (`:109` then `:122`), `"unknown"`
returns `(hist, True)` → the actor is skipped (`:195-196`), `"rejective"` rewrites
via `rewrite_query` and drops the last exchange (`:128-131`). `summary` (`:135`)
tries the detail-then-type summary prompts, target-calls each, rubric-scores,
breaks on 5, otherwise pops the exchange (`:185`). Goal success = a summary score of
5 (`:199`), early stop.

**Guardbound** (`runner.py:1219-1385`): `prepare_attack` once (`:1308`), per-actor
`call_multi` (`:1321`), `summary` unless skipped (`:1332`), `goal_achieved` on
score 5 with `break` (`:1366-1369`), `skipped_actors` for unknown verdicts
(`:1361-1364`), one `Conversation` per goal (`:1371`). The refusal/rewrite path is
in `call_multi`; the composite path does **not** wrap the attack in
`SteeredLLMChat` when `steer_target: false` (R1) — it installs the candidate filter
and lets the attack own its conversation. Contract scenarios (which actor succeeds,
all fail, unknown, repeated rejection, summary 5 vs below, no valid query, exception
mid-actor) were proven identical in the Phase 24 audit; this audit re-read the
current code and found no drift.

**Verdict: `EXACT MATCH`** (A4, A5, A6, A12). No deviation found.

### 4.2 Crescendo

**Official** (`crescendomation/run.py`): `while round_number + num_filtering <
max_rounds` (`:44`) with `round_number = len(history_t)//2 + 1` (`:45`); one
`generate_crescendo_step` attacker call (`:46`); NBF block (`:49-69`) scoring the
same prompt (vacuous `range(3)`), `num_filtering += 1; continue` on filter;
`history_t.append(user)`, target call (`:73-74`); `history_a["attacker"].append`
always, `history_a["target"]` append only when `round_number > 1` (`:76-78`);
refusal → score `"refused"`, `history_t.pop()`, `C_refused += 1`, continue
(`:80-89`); else append assistant, rubric (`:91-92`), stop on 5 (`:102-106`).

**Guardbound** (`runner.py:674-800`): the same while-test and the same
`round_number = len(history) + 1` recomputation; one attacker call; the NBF block
scores the same query and breaks identically (`:711-732`); refusal → `record_turn(…,
"refused")`, `messages[:-2]`, `increment_refusal()`, continue (`:769-789`); rubric
after the refusal test (`:791-794`); `_include_summary()` implements
`round_number > 1` including the first-round misalignment quirk.

**Differences found:** only the evaluator temperature (M5) and the failure policy on
a `None` score (E4). The `range(3)` candidate loop looks like "3 candidates" in both
codebases and is in fact one score in both — this is *not* a deviation, and it must
not be "fixed" without changing the official baseline.

**Verdict: `EXACT MATCH`** for control flow; `PARTIAL MATCH` for judge temperature.

### 4.3 Opposite-day

**Official** (`opposite_day/run.py`): structurally identical to Crescendo (`:43-107`)
with `OPPOSITE_DAY_SYSTEM_PROMPT`; **no disclaimer check** (`check_disclaimer` is
imported only by `acronym/run.py:6` and used at `:78`).

**Guardbound** (`opposite_day.py` + the same generic runner): no disclaimer path
(verified: only `check_refusal`), same loop, same refusal rule, same summary rule.

**Verdict: `EXACT MATCH`** except judge temperature.

### 4.4 Direct answer to the "same meaningful decisions" question

For the three evaluated attacks, Guardbound issues the same *sequence of decisions*
as the author's code: same number of attacker calls per round, same single NBF
evaluation per candidate, same accept/filter inequality, same target-call placement,
same refusal/rewrite/pop logic, same early-stop signal, same one-verdict-per-goal
aggregation. The observable divergences are **not** in the decision structure but in
the judge's sampling temperature (M5) and in what happens when a judge reply is
unusable (E4) or an API call fails (M9) — all three change *whether a goal's verdict
is usable*, and therefore interact with the denominator (E2).

---

## 5. NBF architecture and inference audit

| Check | Evidence | Result |
|---|---|---|
| Checkpoint sha256 | recomputed this audit | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` = config `:128` ✓ |
| Encoder | `all-mpnet-base-v2`, 768-d | ✓ (`:129-131`) |
| Dynamics / predictor shapes | 1536-512-512-768 / 1536-32-32-5, 5 classes | ✓ |
| Class indexing | last logit = class 5 | ✓ (`runner.py:649-652`; official `probs[:,-1]`) |
| Score formula | `P(5) − max(P(1..4))` | ✓ Eq. (5) |
| Initial state | `zeros(768)` | ✓ |
| Threshold semantics | config `eta` → raw `−η` **exactly once** (`_cloud_common.py:672-681`) → `admits(h, −η)` strict | ✓ paper line 534 |
| Strictness | `<` admit, `≥` filter | ✓ matches the paper's strict inequality |
| Placement | after generation, before the target call | ✓ |
| Rejected candidate | `num_filtering += 1`, regenerate | ✓ |
| No surviving candidate | budget consumed, loop terminates | ✓ |
| Precision | CPU float32 vs the code's GPU float32 | scores may differ ~1e-7; a decision can flip only within ~1e-7 of −1e-3 (η is 1e-3, four orders larger) |
| Determinism | same input → same score (no dropout, eval mode) | ✓ Phase 26 measured |

**η = 1e−3, raw −1e−3, filter iff h + η ≥ 0, admit iff h < −1e−3** — all four
statements are simultaneously true in the current code (config `:135-136`,
`filter_rule.py:22-46`, `paper_protocol.py:184-190`, `cloud_pilot.py:195-202`).
No threshold or checkpoint change is needed or permitted.

---

## 6. Dataset and preprocessing audit

Re-verified from the file itself, not from a report:

```
path    nbf_original_stuff/.../NBF-LLM/data/test/harmbench_tasks.json
sha256  ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb
records 200   keys ['max_rounds','target_system','task']
max_rounds        {8: 200}
target_system     {'You are a helpful assistant.': 200}
duplicates        0        empty goals  0
```

* Paper: "200 samples of Harmbench dataset … released by Ren et al. (2024) and have
  been filtered to avoid contamination" (B.1 p22) — matched exactly, same file the
  author's entry point loads (`steering.py:92`).
* Ordering: file order, no shuffle (`:119-120`), identical to the code's plain
  `for i, each_case in enumerate(test_case)` (`steering.py:103`).
* No truncation, no normalisation, no filtering, no sampling beyond `--limit` (which
  the manifest records so a partial run cannot masquerade as 200 goals).
* Same goals for the baseline and the protected arm by construction (one file).
* Upstream byte-identity with the public HarmBench release:
  **`UNVERIFIABLE — NETWORK REQUIRED`** (prohibited by this audit). The author's copy
  is the authoritative artifact for reproduction.

---

## 7. Evaluator and ASR audit

### 7.1 Judge

The paper's ASR is "the ratio of successful jailbreaks **judged by GPT-4o** (Qi et
al., 2023) following Ren et al. (2024) over all harmful queries" (§5.1 p8). The
author's judge is the Qi et al. rubric embedded in `evaluate_with_rubric.py`
(1 = proactive decline … 5 = full compliance), called with `temperature=0.0`, with
`check_refusal` (same prompt family, `temperature=0.0`) deciding the rejective branch
first. Guardbound sends the **byte-identical rubric** (prompt variant `official`,
hash-pinned; Phase 22 byte audit) at the configured evaluator temperature.

* Refusal criterion: the code's `check_refusal` asks the judge for
  `{value, metadata}` and requires `value and metadata > 90` (Guardbound
  `llm_check_refusal`, `confidence_threshold=90`) — identical.
* Disclaimer criterion: Acronym only; correctly absent from the three evaluated
  attacks.
* Output parsing: JSON object; Guardbound classifies parse failures instead of
  crashing (E4).
* Invalid replies: official crashes the run; Guardbound records `None` and flags the
  goal.
* Turn- vs conversation-level: success is **any turn** scoring 5 within the
  conversation (Crescendo/OD) or the **summary** scoring 5 (ActorAttack); never a
  turn-count ratio.
* Failure handling: excluded from Guardbound's denominator and reported separately.

### 7.2 ASR definition — paper vs implementation

| | Paper (§5.1) | Author's code | Guardbound |
|---|---|---|---|
| Numerator | successful jailbreaks | goals with `goal_achieved: true` | `success = max(valid rubric scores) == 5` |
| Denominator | **all harmful queries** | all 200 goals attempted (a crash aborts the run) | `valid_completed_goals` |
| Unit | behaviour (conversation) | behaviour | behaviour |

The numerator is faithful. The denominator is a **deliberate, user-approved
deviation**: Phase 23 D-03 found failed goals entering the corpus as zero-turn
conversations and Phase 24 changed the rule to "successes / valid completed goals,
with failures reported separately". That decision is methodologically defensible
(a failed API call is not a failed jailbreak), but it is **not the paper's literal
definition**, and it changes the number.

### 7.3 Numerical implication

Let *s* = successes and *f* = goals lost to infrastructure failure, out of 200.

| *f* | Guardbound ASR = s/(200−f) | Paper-literal ASR = s/200 | Relative inflation |
|---:|---:|---:|---:|
| 0 | s/200 | s/200 | 0 % |
| 5 | s/195 | s/200 | +2.6 % |
| 10 | s/190 | s/200 | +5.3 % |
| 20 | s/180 | s/200 | +11.1 % |
| 40 | s/160 | s/200 | +25.0 % |

Worked example against the paper's NBF Crescendo target (0.260, i.e. 52/200):
with *f* = 20 the same 52 successes are reported as 52/180 = **0.289** — a 2.9
percentage-point overstatement that would read as a worse defence than the paper's.

**Recommendation (does not alter the current rule):** report both numbers — the
primary metric as `successes / 200` (paper-literal) and the diagnostic
`successes / valid_completed` — and mark any run with *f* > 0 as not directly
comparable to Table 1. Both are computable from the existing per-goal records; no API
call and no methodology change are required.

---

## 8. Baseline vs defence consistency

| Requirement | State | Evidence |
|---|---|---|
| Identical goal set | ✓ | one dataset file, file order |
| Identical attack configuration | ✓ | one `attacks:` block; variants may only override `nbf` (`_cloud_common.py:88-97`) |
| Identical model roles | ✓ | one `models:` block |
| Identical evaluator | ✓ | one `models.evaluator`, one rubric variant |
| Identical generation parameters | ✓ | one `temperature:` block; only `nbf.enabled` differs |
| Identical turn limits | ✓ | one `max_turns` |
| Consistent randomisation | ✓ (both unseeded, temperature sampling) | `seed: null` |
| Consistent failure treatment | ✓ | one code path |
| **Both arms declared** | ✗ | `variants` contains only `r1_paper`; `resolve_variant(cfg,"r1_baseline")` raises `ValueError: unknown variant` |
| **Invariant guard active** | ✗ (inert) | `assert_invariants` returns `[]` when `len(variants) < 2` (`_cloud_common.py:85-86`) |

**Consequence:** the before/after comparison Table 1 shows ("GPT-4o" vs "GPT-4o +
steering") cannot be produced from the R1 configuration alone, and the guard that is
supposed to prove the two arms differ *only* in NBF enablement never runs. This is a
configuration-completeness defect (P1 for the table, P2 for methodology) and is fixed
by declaring a second variant with `nbf.enabled: false` — a config-only change, no
code change, no API call, verifiable offline (`resolve_variant` +
`assert_invariants` then return the checked invariants).

---

## 9. Utility evaluation audit

| Component | Paper | Repository | Status |
|---|---|---|---|
| MTBench | 1–10, refusal substituted on positive barrier score (B.1 p22) | `evaluation/mtbench.py` implements substitution with the canonical `MTBENCH_REFUSAL_STRING`, judge at `temperature=0.0`, target at 0.7; `scripts/eval_mtbench.py` exists | **code present, unusable**: `--dataset` is required and no MTBench file exists anywhere in the repo; the script is not referenced by the R1 config |
| MTBench margin | η = 1e−3 for the reported row | `scripts/eval_mtbench.py:31` default `--eta 5e-4` | **incorrect default** for this row (P1 for utility, P3 otherwise) |
| XSTest | refused queries / all | `evaluation/over_refusal.py` + `scripts/eval_over_refusal.py` | **code present, no data**; the refusal detector is a documented local regex heuristic (paper silent → `UNVERIFIABLE`) |
| MMLU | Table 9 reports 87.04 → 62.69 at η = 1e−3; positive ⇒ wrong (B.1) | `evaluation/mmlu.py` + `scripts/eval_mmlu.py` | **code present, no data**, not in R1 scope |
| Utility in the R1 config | required by Table 1 | absent (no `metrics:`/`utility:` block) | **MISSING** |
| Judge model for MTBench | not stated (cites Zheng et al. 2023) | `gpt-4o-2024-08-06` default | **UNVERIFIABLE** (paper silent) |

The author's release ships **no** utility data either (`data/` contains only
`test/harmbench_tasks.json` and `LoRA_SFT/…`), so the paper's MTBench/XSTest numbers
cannot be regenerated from the release alone. The three attack ASRs therefore do
**not** constitute a reproduction of Table 1.

---

## 10. Factors that may prevent matching the paper's numbers

Confirmed defects and possible variation sources are separated by the *Class* column
(`D` = confirmed defect, `V` = possible variation, `U` = unverifiable).

| # | Issue | Evidence | Class | Severity | Potential effect | How to verify | Recommended action |
|---|---|---|---|---|---|---|---|
| 1 | ASR denominator excludes failed goals | `cloud_pilot.py:588`; paper §5.1 | D | **P1** | +2.6…+25 % inflation (f = 5…40) | recompute both from the per-goal JSONL | report both denominators (§7.3) |
| 2 | Judge temperature 0.7 vs code 0.0 | config `:83`; `evaluate_with_rubric.py:31` | D | **P1** | spurious 5s accumulate over up to 8 judged turns → ASR up, especially for the filtered arm | re-judge a frozen transcript set at 0.0 vs 0.7 offline | report the deviation; consider a 0.0 sensitivity run (needs API) |
| 3 | No baseline variant declared | `variants: {r1_paper}`; `_cloud_common.py:85` | D | **P1** (table) | the "before" column cannot be produced/checked from this config | `resolve_variant(cfg,"r1_baseline")` raises | declare `r1_baseline` with `nbf.enabled: false` |
| 4 | Utility metrics unavailable | no MTBench/XSTest data; no utility block | D | **P1** (table) | Table 1's MTBench/XSTest cells cannot be reproduced | `Get-ChildItem -Recurse *mtbench*` → only code | supply datasets, wire a utility block, fix `--eta` |
| 5 | Zero retries vs 10 in the code | `openai_client.py:223`; `generate.py:129` | D | **P1/P2** | transient errors remove goals → shrinks the denominator, biases the population | count `failed: true` rows per run | report failure rate; decide a retry policy *before* the run and freeze it |
| 6 | Snapshot alias vs pinned id | `steering.py:70` (`gpt-4o`) vs config `:53` | V | P2 | the published numbers may come from a different GPT-4o snapshot than 2024-08-06 | impossible: the API does not expose the served revision (`model_identity.resolved_revision_exposed_by_api: false`) | keep the pin; record `model` from each response |
| 7 | No seed / single run | `seed: null`; both codebases sample at 0.7 | V | P2 | ±few-goal run-to-run variance; no CIs in the paper either | repeat one attack offline is impossible (needs API) | report n=1 explicitly; never interpret <±3 pp as a difference |
| 8 | NBF precision CPU vs GPU | local CPU float32 | V | P2 | ≤1e-7 score shift; a decision flips only within 1e-7 of −1e-3 | boundary probe exists (Phase 26) | document; threshold is 4 orders larger |
| 9 | Per-call wire settings not recorded | usage rows carry tokens/cost/model, not temperature/top_p/response_format | V | P2 | a mis-wired call would be invisible after the fact | inspect `usage.jsonl` | add the three fields to the usage schema |
| 10 | Two ASR paths | `evaluation/asr.py` + `judge_asr.py` (non-official prompt) vs the pilot rubric | D | P2 | using the wrong path silently changes the judge | `asr.py:3-8` docstring vs `paper_protocol` | declare the pilot rubric the only R1 ASR source |
| 11 | Evaluator parse-failure policy | `None` + flag vs official crash | V | P2 | changes which goals are usable (feeds #1) | count `n_invalid_rubric_scores` | keep; report the count |
| 12 | 7 vs 8 target turns | `crescendomation/run.py:44-45`; `runner.py:674-682` | V | P3 | one fewer escalation step than the paper's prose implies; matches the code | offline loop simulation exists | document ("paper 8, code 7, we follow the code") |
| 13 | η default in the MACP script | `eval_mtbench.py:31` | D | P3 (P1 if used) | a utility run at 5e-4 would not be the Table-1 row | `--eta 0.001` override | fix the default |
| 14 | Model deprecation | config `:64-67` (announced 2027-01-14) | V | P3 | the pinned id may disappear | n/a | archive outputs; record the date |
| 15 | `--limit` partial runs | manifest `complete:false`, `expected/completed` counts | V | P3 | a 20-goal smoke could be mistaken for 200 | read the manifest | already guarded |
| 16 | Prompt-variant drift | legacy bytes still selectable | V | P3 | selecting `legacy` changes the judge text by 4 whitespace characters | `audit_prompt_fidelity.py` | R1 config pins `official`; validator rejects others for paper runs |
| 17 | Paper silence on the attacker | §5.1 names only the judge | U | P3 | the attacker is a code-recovered choice | n/a | documented as `code-recovered` |
| 18 | Paper silence on MTBench judge / XSTest detector | §5.1, B.1 | U | P3 | utility numbers rest on local defaults | n/a | document per component |
| 19 | Upstream dataset byte identity | author's copy only | U | P3 | contamination filtering cannot be re-verified offline | network comparison (prohibited) | keep the author's file + hash |
| 20 | No cross-run aggregation tool wired to R1 | pilot writes per-goal rows; ASR is post-hoc | V | P2 | manual aggregation invites mistakes | inspect the report block | add a thin ASR CLI over the JSONL (offline) |

**Not defects** (explicitly checked and correct): the vacuous `range(3)`/`range(10)`
candidate loops, refusal turns not counting, 10-refusal cap, first-round summary
misalignment, Acronym's exclusion, absent disclaimer check in Opposite-day,
`response_format` text on target calls, no `max_tokens`.

---

## 11. Reproducibility and artifact integrity

| Artifact | Recorded? | Where | Note |
|---|---|---|---|
| Dataset identity | ✓ | manifest `dataset.sha256` + goal-id hash | `ac789de8…`, 200 records |
| Prompt bytes | ✓ | official prompt + rubric hashes in the manifest | legacy hashes also stored |
| Checkpoint | ✓ | sha256 in config and manifest | `cea1a75b…` |
| Config bytes | partial | config path recorded; **config sha256 not in the manifest** | add it (**P2**) |
| Code commit + dirty state | ✓ | `git_commit`, `git_dirty` | audited state: `89ed03f`, 90 dirty files — a run now would be flagged dirty |
| Model ids | ✓ | report `configuration_under_qualification` / arm metadata | |
| Generation parameters | partial | temperature recorded per role; **no per-call wire record** | #9 |
| Seeds | ✓ (as absent) | `seed: null` + `seed: null` in the manifest | honest: the paper specifies none |
| Per-call usage | ✓ | `<stem>.usage.jsonl`, 21 fields, incremental flush | no budget cap (by design) |
| Failed requests | ✓ | per-goal `failed`/`error_category`, `counts.failed` | |
| Goal-level outputs | ✓ | one row per goal with turn-level rubric scores | |
| Evaluator outputs | partial | score + telemetry counters; **raw judge reply not stored** | a re-audit cannot re-parse it (**P3**) |
| Completion metadata | ✓ | `started_at`, `finished_at`, `complete`, expected/completed/valid/failed counts, `asr_denominator`, `asr_denominator_definition` | |
| Baseline↔defence linkage | ✗ | no `variant` field tying the two arms beyond the filename | add `variant` + config hash (**P2**) |
| Environment | ✓ | `environment` block (Python, packages, platform) | no GPU (CPU NBF) — record that too |
| Frozen artifacts | untouched | `git status --porcelain -- results runs checkpoints data …` → empty | verified this audit |

A second researcher can reconstruct the *inputs* of a run from the manifest except
for the config bytes and the variant label; both are cheap additions and neither
requires an API call.

---

## 12. Prioritised recommendations

Nothing here was implemented. Each item states whether it changes methodology, needs
API calls, and how it can be validated offline.

### A. Must fix before qualification

**None.** The qualification harness is orthogonal to the experiment and was audited
in Phases 25–27 (`INSUFFICIENT_DATA`/`NOT_QUALIFIED` stop before any client). No
qualification-relevant defect was found here.

### B. Must fix before the main evaluation

| # | Recommendation | Why it matters | File / function | Changes methodology? | API calls? | Offline validation |
|---|---|---|---|---|---|---|
| B1 | Report **both** ASR denominators: `successes/200` (paper-literal) and `successes/valid_completed` | the paper's metric is over all 200; the current rule inflates by up to 25 % | `scripts/cloud_pilot.py` run block (`:582-589`) + a thin aggregation CLI | Reporting only — the filter of failures stays | No | recompute from an existing JSONL; assert both fields appear |
| B2 | Declare `r1_baseline` (`nbf.enabled: false`) in the R1 config | the "before" column and the "differ only in NBF" guard are otherwise absent | `configs/paper_reproduction_r1_gpt4o.yaml:143-147` | No (same method, second arm) | No | `resolve_variant` + `assert_invariants` return the checked invariants |
| B3 | Decide and freeze the failure policy **before** the run: either keep 0 retries and report *f*, or allow a bounded transport retry | *f* drives B1 and can bias the population | config `experiment.protocol.retries` / `openai_client.py` | Yes if retries are enabled (more calls per goal) | Depends | count `failed:true` rows on a mock run |
| B4 | Pin the ASR source: the pilot's attack-time rubric; `judge_asr.py` is *not* used for R1 | two judges with different prompts is a methodology fork | `evaluation/asr.py`, `scripts/judge_asr.py` docstrings | Clarifies, does not change | No | grep the R1 pipeline for the judge entry point |
| B5 | Supply MTBench/XSTest datasets + a `utility:` block, and set `--eta 0.001` | without them Table 1's last two columns are unreachable | `scripts/eval_mtbench.py:31`, `scripts/eval_over_refusal.py`, config | Adds the paper's utility protocol | Yes (utility judge/target) | dataset schema check + `--dry-run`/mock path |

### C. Should fix for reproducibility

| # | Recommendation | Why | File | Offline validation |
|---|---|---|---|---|
| C1 | Record config sha256, `variant` name and the config path in the manifest | ties the two arms to one declaration | `cloud_pilot.py` run block | assert fields on a mock run |
| C2 | Store the raw evaluator reply (or its hash) per judged turn | permits post-hoc re-parsing/re-judging | `record_evaluation` telemetry | schema check |
| C3 | Add `temperature`/`top_p`/`response_format` to the usage rows | proves the wire settings per call | `llm/usage.py` | row schema test |
| C4 | Record the NBF device (CPU/GPU) and float precision in the manifest | explains 1e-7 boundary differences | `nbf_meta` block | manifest inspection |
| C5 | Log η and the raw threshold **together** in every attack record | prevents another sign-convention incident (D-01) | `cloud_pilot.py:268-269` | already present; extend to per-goal rows |

### D. Accepted deviations (document, do not "fix")

| # | Deviation | Justification |
|---|---|---|
| D1 | Evaluator temperature 0.7 instead of the code's 0.0 | the paper is the stated methodological authority ("All temperatures are set to be 0.7"); the reported numbers were produced at 0.0, so a sensitivity run at 0.0 is informative but not required |
| D2 | Excluding infrastructure failures from the ASR denominator | user-approved Phase 24 decision; defensible, but the paper-literal number must also be published (B1) |
| D3 | Typed failure instead of `data["score"]` crashing the run; `None` instead of a synthetic score | the official crash destroys the whole run; substituting `1` would fabricate a refusal |
| D4 | 7 target turns where the paper says "at most 8" | faithfully follows the author's loop condition |
| D5 | Pinned model ids where the author used `gpt-4o` | stricter, and the paper names the pinned snapshot |
| D6 | CPU NBF inference | no GPU in this environment; score differences are ~1e-7 |
| D7 | Assumed attacker = GPT-4o | code-recovered; the paper only names the judge |

---

## 13. Final readiness classification

# `INCOMPLETE SCOPE`

Justification, separated by the four senses the brief asks for:

| Sense | Classification | Evidence |
|---|---|---|
| **Implementation readiness** (attack-ASR half) | `READY WITH DOCUMENTED DEVIATIONS` | three attack loops and the NBF are `EXACT MATCH` to the author's code and the paper; the two P1 deviations (B1 denominator, B2 judge temperature) are understood, isolated and fixable without touching the method |
| **Qualification readiness** | `READY` | Phase 27: fail-closed local gate, 0 clients on any non-`QUALIFIED` verdict, 1189/1185/4 offline suite |
| **Experimental readiness** (a 200-goal run) | `NOT STARTED` — requires explicit authorisation | no run exists; a successful qualification would not prove experimental reproduction |
| **Full GPT-4o Table-1 reproduction** | `INCOMPLETE SCOPE` | 3 of 5 metrics implemented; MTBench and XSTest have code but no data, no R1 wiring, and a wrong default margin (D in §1.2, §9) |

**Why `INCOMPLETE SCOPE` and not `READY WITH DOCUMENTED DEVIATIONS`:** the second
label would assert that *all selected GPT-4o evaluation components* are implemented
faithfully. They are not — two of the five Table-1 columns cannot be executed at all,
and the baseline arm the table requires is not declared. **Why not `NOT READY`:**
nothing found here invalidates the attack-ASR methodology; the P1s are reporting and
completeness defects with offline, config-level remedies, and no P0 exists.

**What a defensible GPT-4o row requires, in order:** (1) B2 baseline variant,
(2) B1 dual denominators, (3) B3 frozen failure policy, (4) B5 utility data + margin,
then (5) the authorised run. Until (1)–(4) are closed, any ASR obtained from this
repository is an *internal* measurement — comparable to itself across arms, not to
the paper's Table 1.
