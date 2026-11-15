# PHASE 32 — UTILITY BENCHMARK METHODOLOGY RESOLUTION & DATASET ACQUISITION PLAN

**MT-Bench methodology: `PARTIALLY RESOLVED`**
**XSTest methodology: `PARTIALLY RESOLVED`**
**MT-Bench dataset provenance: `PARTIALLY VERIFIED`**
**XSTest dataset provenance: `PARTIALLY VERIFIED`**
**Dataset acquisition plan: `READY`**
**Utility implementation: `NEEDS CHANGES`**
**Full Table 1: `NOT READY`**

Evidence-gathering and methodology-resolution phase only. **No implementation change, no
experiment, no dataset acquisition.** No scripts, configs, tests or frozen artifacts were
modified; one report file was created. No dataset body was retrieved in this phase.

---

## 1. Executive summary

The blocking questions were "what exactly is the utility protocol, and can it be pinned?" Both
benchmarks are now specified far enough to execute — **not** far enough to claim paper-exact
reproduction, because the NBF paper specifies almost nothing about them.

| Question | Answer |
|---|---|
| MT-Bench release used by the authors? | **Not stated anywhere.** Pin chosen: FastChat `v0.2.36` / dataset commit `b494d0c6…` |
| MT-Bench protocol recoverable? | **Yes, from the benchmark publication + FastChat code**: 80 questions × 2 turns, GPT-4 single-answer grading, 1–10 per turn, **average of 160 = 80 × 2 scores** |
| Does the paper corroborate per-turn scoring? | **Yes** — its own Fig. 7 plots "MTBench Helpfulness, Turn 1" and "Turn 2" |
| MT-Bench temperature? | **Conflict**: paper says "all temperatures 0.7"; upstream uses per-category 0/0.1/0.7 (plus a `--force-temperature` escape hatch) |
| MT-Bench judge? | **Not specified by the paper.** Upstream default GPT-4; Guardbound defaults to GPT-4o |
| XSTest denominator? | **Not specified by the paper** ("refused queries over all"). The benchmark itself reports **safe prompts n=250** and unsafe n=200 separately |
| XSTest detector? | **Not specified by the paper.** Upstream ships two detectors (string-match, GPT-4 3-class); Guardbound uses a third (regex) |
| Dataset revisions pinned? | **Yes** — commit + git blob SHA-1 for every required file; paper correspondence still unverified |
| Can Table 1 be produced today? | **No** — data absent, and the implementation has defects (§7) |

Three findings deserve emphasis:

1. **Guardbound's MTBench aggregate is wrong.** `evaluate_mtbench` computes
   `mean_score = sum(turn-1 scores)/n` (`mtbench.py:357-359`), i.e. `mean_score` is a duplicate of
   `turn1_mean` and silently discards turn 2. The reported Table-1 column would not be the
   benchmark's MT-Bench score even with the right data.
2. **Guardbound's steered over-refusal run is not steered.** `eval_over_refusal.py` loads the
   barrier (`:96-97`) and then never uses it, so a `--barrier-checkpoint` run produces the same
   numbers as the baseline while labelling itself `guardbound`.
3. **The paper's own XSTest numbers are internally inconsistent** (Table 3 vs Table 13, §6.2), and
   the XSTest publication itself has a 250/200 slip in §4.3. Exact reconstruction of the paper's
   XSTest column is therefore impossible from the paper alone.

A Phase 31 provenance error was also corrected: the XSTest `readme.md` blob recorded there
(`2850b164…`, 2 772 B) is the **`main`** version; at the pinned commit it is `92bfa9c9…`, 2 535 B.

---

## 2. Scope and actions performed

Read-only actions:

1. Extracted the full text of the local paper (`2503.00187v3.pdf`, 31 pages, 101 854 chars) and
   located every MTBench / XSTest / over-refusal / temperature / judge statement.
2. Downloaded the two **benchmark publications** (XSTest NAACL 2024 PDF, MT-Bench arXiv PDF) to
   the session scratchpad and extracted the metric and prompt definitions. These are papers, not
   datasets; they were not copied into the repository.
3. Inspected the author's local repository, including its git metadata, and searched it for any
   utility code, data or refusal-substitution string.
4. Inspected the pinned upstream **code and documentation** for both benchmarks (FastChat
   `llm_judge`; XSTest evaluation scripts + README).
5. Retrieved upstream **directory-tree metadata** (paths, blob SHA-1, byte size) through the GitHub
   git-trees API. No dataset body was requested or received.
6. Audited Guardbound's utility implementation, config, readiness schema and CLIs, without editing
   them.
7. Re-hashed the frozen artifacts and ran five selected offline readiness/provenance tests.

Not done: no dataset download, clone, copy, materialisation or modification; no MT-Bench/XSTest
run; no model or judge inference; no paid API call; no qualification, smoke test, pilot or
experiment; no change to any script, config, test, attack, NBF component or arm.

---

## 3. Sources inspected and exact versions

| # | Source | Version / identifier | What was read |
|---|---|---|---|
| S1 | Paper | `2503.00187v3.pdf`, TMLR 02/2026, 31 pp. | full text; §5.1 p8, Table 1 p8, Table 3 p9, Fig. 7 p24–25, B.1 p22, Tables 10/13/15 |
| S2 | Author repo | `https://github.com/HanjiangHu/NBF-LLM.git` @ **`ac014b2cc09679dca087d5f7cf2ec9d3cf602e0c`** (2026-06-24, "update tmlr"); local clone HEAD and published HEAD match; working tree clean | file tree, `steering.py`, `train.py`, attack code; searched for utility/refusal strings |
| S3 | Author supplementary code | `TMLR_supp_code/`, 23 files | same scope as S2: attacks, `train.py`, `steering.py`, HarmBench + SFT data |
| S4 | FastChat | tag **`v0.2.36`** = commit `b21d0f780ca4472a13714262a0790f2ee1ade659` (2024-02-11, "Update version to 0.2.36 (#3033)"); MT-Bench data commit **`b494d0c6b4e7935f1764f8439e75da3e66beccc7`** (2023-06-16) | `common.py`, `gen_model_answer.py`, `gen_api_answer.py`, `gen_judgment.py`, `show_result.py`, `llm_judge/README.md`; `data/` tree metadata |
| S5 | MT-Bench publication | Zheng et al. 2023, arXiv:2306.05685 | §3.1, §4.1, §5, Figure 6 prompt text |
| S6 | XSTest repo | commit **`475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5`** (2025-02-11, "overdue cleanup"); no tags, no releases; CC-BY-4.0 | `readme.md`, `classify_completions_strmatch.py`, `classify_completions_gpt.py`; full tree metadata |
| S7 | XSTest publication | Röttger et al., NAACL 2024, pp. 5377–5400 | abstract, Table 1, Table 2, §4.2–4.3, Appendix D taxonomy |
| S8 | Guardbound | working tree at `main` @ `89ed03f` + uncommitted Phase 22–31 work | `evaluation/mtbench.py`, `evaluation/over_refusal.py`, `evaluation/refusal_detector.py`, `evaluation/reporting.py`, `evaluation/readiness.py`, `defense/post_processing.py`, `defense/filter_rule.py`, `defense/steered_chat.py`, `scripts/eval_mtbench.py`, `scripts/eval_over_refusal.py`, `scripts/run_full_evaluation.py`, `configs/paper_reproduction_r1_gpt4o.yaml` |
| S9 | Frozen artifacts | local files | SHA-256 re-verified (§9) |

Exact upstream file identities (git blob SHA-1, byte size) at the pinned revisions:

| File | Blob SHA-1 | Bytes |
|---|---|---|
`fastchat/llm_judge/data/mt_bench/question.jsonl` | `464e2c22f7a021a0f9a584d232338a9a17c40066` | 48 929 |
`fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl` | `8b66387a3129cbb117178fe515491df553dd52df` | 51 109 |
`fastchat/llm_judge/data/judge_prompts.jsonl` (harness, not dataset) | `d1e499f12d18c57338d883d43f5735e512c00d8c` | 10 457 |
`fastchat/llm_judge/common.py` | `fc223573cbf3ca2633dc022e50dca1e1d180a95f` | 18 459 |
`fastchat/llm_judge/gen_model_answer.py` | `5468dce0688933d3ccae6f3c632676d9905ff28d` | 6 809 |
`fastchat/llm_judge/gen_api_answer.py` | `ed794802a06bf6a02acb7a1a966896bdf27b5cbc` | 4 147 |
`fastchat/llm_judge/gen_judgment.py` | `3d952c8b09459f64397cb23f2f56e121f024b0a4` | 9 575 |
`fastchat/llm_judge/show_result.py` | `55ceb9e14483c2873a85aca2e1310c97d2d7cac7` | 3 744 |
`fastchat/llm_judge/README.md` | `b55b2af1eddf7aa8bc5141fad98dfac5407baba9` | 5 576 |
`xstest_prompts.csv` | `ad38204d70b3964387f9690169d11afbc4756d01` | 38 719 |
`readme.md` (XSTest, at pinned commit) | `92bfa9c944ad3c5bd649d00bde2c33e90fcd238a` | 2 535 |
`evaluation/classify_completions_strmatch.py` | `aa8fa43a2043c7d400923bd4ffb628154aca1b7c` | 1 368 |
`evaluation/classify_completions_gpt.py` | `b9f911978a4ee5c91aa65239a59c2b979cd61576` | 2 721 |
`evaluation/analysis.ipynb` | `c24df55ee50fc6b0dd3d4a9990380bff1050accd` | 136 168 |

`judge_prompts.jsonl` is the only file whose **body** would be needed for byte-exact judge-prompt
reproduction; only its metadata was read.

---

## 4. MT-Bench methodology audit

### 4.1 What the NBF paper actually specifies

| Location | Statement |
|---|---|
| p8 §5.1 | "... MTBench (Zheng et al., 2023), where ... **MTBench specifically evaluates an LLM's ability to handle multi-turn conversations in dialogue scenarios with scores from 1 to 10**." |
| p8 §5.1 | "**All temperatures are set to be 0.7.**" (sentence sits in the ASR/helpfulness paragraph; its scope — target only, or target + judge — is not qualified) |
| p8 Table 1 | GPT-4o: MTBench **9.35**; GPT-4o + steering: **8.77** |
| p9 Table 3 | llama-3-8b-instruct MTBench 7.96 / 8.13 / 7.52 / 7.90 / 7.42; Phi-4 8.23 / 8.59 / 8.06 / 8.18 / 7.76 |
| p24–25 Fig. 7 | axes labelled "MTBench Helpfulness, **Turn 1**" and "MTBench Helpfulness, **Turn 2**" |
| p22 B.1 | "For MTBench evaluation, when the neural barrier function predicts positive values, we replace the "non-safe" answer with *"I’m sorry, but I cannot provide information on that. It is important to prioritize safety and ethical considerations in our conversation. If you have any other questions on a different topic, feel free to ask!"*, **followed by regular evaluation of MTBench**." |

**Nothing else.** The paper never states: the MT-Bench release/commit, the question count, the
number of turns, the judge model, the judge prompt, judge decoding, reference-answer use, the
aggregation rule, or whether the full set is used. Those are **NOT SPECIFIED**, not "defaults we
may assume".

### 4.2 What the benchmark itself defines (evidence hierarchy level 4 → 3)

* **80 questions, 2 turns each.** MT-Bench paper §4.1: "We generate answers for all 80 questions";
  FastChat `README.md`: "total_num_questions": 80.
* **Both turns scored; overall score is the mean of 160 turn-scores.** MT-Bench paper §5: "We ask
  GPT-4 to give a score for each turn on a scale of 10 ... and report an **average score of
  160 = 80 × 2 turns**." FastChat `show_result.py::display_result_single` prints the turn-1 mean,
  the turn-2 mean, and `df[["model","score"]].groupby("model").mean()` — the pooled mean.
* **Judge = GPT-4, single-answer grading.** `gen_judgment.py` default `--judge-model gpt-4`,
  mode `single` uses `single-v1` / `single-math-v1` / `single-v1-multi-turn` /
  `single-math-v1-multi-turn`.
* **Judge protocol for the single path**: `common.py::run_judge_single` →
  `chat_compeletion_openai(model, conv, temperature=0, max_tokens=1024)`; output format
  `[[rating]]`; a parse failure yields `rating = -1`. (`max_tokens=2048` belongs to
  `run_judge_pair`, *not* to the single-answer path used for MT-Bench scores — this corrects
  Phase 30.)
* **Judge prompt text** is published in the benchmark paper, Figure 6 (single-answer):
  "Please act as an impartial judge and evaluate the quality of the response provided by an AI
  assistant to the user question displayed below. ... After providing your explanation, please
  rate the response on a scale of 1 to 10 by strictly following this format: `[[rating]]`, for
  example: `Rating: [[5]]`." A reference-guided multi-turn variant is Figure 10. FastChat's
  non-reference multi-turn template is the `single-v1-multi-turn` entry of
  `judge_prompts.jsonl` (body **not** retrieved).
* **Both turns come from the dataset** (`question["turns"]`); history is preserved by
  `conv.append_message` / `conv.update_last_message`.
* **Generation temperature is per category**: `common.py::temperature_config` =
  `writing 0.7, roleplay 0.7, extraction 0.0, math 0.0, coding 0.0, reasoning 0.0, stem 0.1,
  humanities 0.1`. `gen_api_answer.py` applies it unless `--force-temperature` is given.
* **Generation cap** = `--max-tokens` default **1024** (API path) / `--max-new-token` default 1024
  (local path); `num_choices` default 1.
* **Judge transport retries**: `API_MAX_RETRY = 8`, `API_RETRY_SLEEP = 10` (`common.py`).
* **Provenance caveat:** at the pinned release, `gen_api_answer.py` only accepts
  `gpt-3.5-turbo`, `gpt-4`, `claude-v1`, `claude-instant-v1` and raises otherwise. GPT-4o and o1
  post-date `v0.2.36` (2024-02-11). The authors therefore **cannot** have produced the Table 1
  GPT-4o/o1 rows with this exact file; they used their own harness or a later release. The pin is
  a *protocol reference*, not evidence of the authors' code.

### 4.3 Methodology table

Classification per row: **CONFIRMED** (established, no conflict) · **PARTIALLY CONFIRMED**
(established by the benchmark, not by the paper) · **CONFLICT** (authoritative sources disagree) ·
**NOT SPECIFIED** (no source establishes it) · **UNVERIFIED** (claimed but unchecked).

| Parameter | Paper specification | Author implementation | Upstream implementation | Guardbound status | Evidence | Resolution |
|---|---|---|---|---|---|---|
| Dataset release | none | none (no MT-Bench asset) | FastChat `v0.2.36`; data commit `b494d0c6…`; `question.jsonl` `464e2c22…`, 48 929 B | loader expects `{question, reference?, turn2_question?, category?}`; `data/raw/mtbench.jsonl` absent | S1 §5.1; S2 tree/grep; S4 tree API | **UNVERIFIED** — paper release unknown; pin is a defensible reference only |
| Questions | none | none | 80 | reads all rows, no count check | S5 §4.1; S4 README | **PARTIALLY CONFIRMED** |
| Turns per question | none | none | 2 (`question["turns"]`) | 2 | S4 `gen_model_answer.py` | **PARTIALLY CONFIRMED** |
| Both turns scored | implied by Fig. 7 | none | yes: 160 = 80 × 2 | yes (two judge calls) | S1 Fig. 7; S5 §5 | **CONFIRMED** |
| History across turns | none | none | yes | yes (`msgs1 + a1 + q2`) | S4 `gen_api_answer.py`; S8 `mtbench.py:303-306` | **PARTIALLY CONFIRMED** |
| Categories | none | none | 8 categories, 10 q each; all used | `category` carried but not used for selection | S4 `common.py` | **PARTIALLY CONFIRMED** |
| Generation temperature | "all temperatures 0.7" | none | per-category 0/0.1/0.7 (`--force-temperature` overrides) | fixed 0.7 (`mtbench.py:296,308`) | S1 p8; S4 `common.py` | **CONFLICT** — paper 0.7 vs upstream per-category; resolvable by `--force-temperature 0.7` |
| Max generation tokens | none | none | 1024 | no cap (`max_output_tokens=None` default) | S4 `gen_api_answer.py`; S8 `openai_client.py` | **NOT SPECIFIED** — set 1024 to match benchmark |
| Stop sequences | none | none | `conv.stop_str` from the model adapter | none | S4 `gen_model_answer.py` | **NOT SPECIFIED** |
| System prompt | none | none | conversation template for the model id (exact text not inspected) | none (user-only messages) | S4 `get_conversation_template`; S8 `mtbench.py:294` | **NOT SPECIFIED** |
| Target model | GPT-3.5-turbo, GPT-4o, o1, Claude 3.5 Sonnet (no snapshots) | none | n/a | `gpt-4o-2024-08-06` | S1 Table 1; S8 config | **PARTIALLY CONFIRMED** — family matches; dated snapshot is a reproduction decision |
| Refusal substitution rule | B.1: replace the "non-safe" answer when barrier is positive, then evaluate | none | n/a | `mtbench_replace_answer(h>0, a2)` on turn 2 only (`mtbench.py:313-327`) | S1 p22; S8 `post_processing.py:51-65` | **PARTIALLY CONFIRMED** — rule confirmed; *which* answer/turn is not stated |
| Substitution trigger | B.1 "positive values" (h > 0) | none | n/a | h > 0 | S1 p22 vs S1 §5.2 (`h + η ≥ 0`) | **CONFLICT** (paper-internal); Guardbound follows the benchmark-specific B.1 rule |
| Substitution string | 209 chars, starts "I’m" (U+2019) | none | n/a | 209 chars, starts "I'm" (U+0027), SHA-256 `ad946dc6…` | S1 p22; S8 `post_processing.py:18-23` | **PARTIALLY CONFIRMED** — identical except one codepoint |
| Judge model | none | none | GPT-4 (paper §5; FastChat default `gpt-4`) | default `gpt-4o-2024-08-06` (`eval_mtbench.py:23`) | S5 §5; S4 `gen_judgment.py`; S8 | **NOT SPECIFIED** — Guardbound deviates from the benchmark |
| Judge prompt | none | none | Figure 6 single-answer text; FastChat `single-v1-multi-turn` (body not retrieved) | local `DEFAULT_MTBENCH_TURN_PROMPT`; `--judge-prompt` is a **no-op** (unreachable branch `mtbench.py:201-205`) | S5 Fig. 6; S4 `common.py`; S8 | **NOT SPECIFIED** + implementation defect |
| Single vs reference-guided | none | none | single-answer; reference-guided for math/reasoning/coding (`NEED_REF_CATS`) | single-turn, never reference-guided, no dialogue context on turn 2 | S4 `common.py`, `gen_judgment.py`; S8 `mtbench.py:334-343` | **NOT SPECIFIED** |
| Reference answers | none | none | `reference_answer/gpt-4.jsonl` for math/reasoning/coding | never used; `reference` is misused as a turn-2 *question* fallback (`mtbench.py:302`) | S4 `common.py`; S8 | **NOT SPECIFIED** + defect |
| Judge temperature | "all temperatures 0.7" (scope unqualified) | none | 0 | 0.0 (`mtbench.py:181,225`) | S1 p8; S4 `run_judge_single`; S8 | **CONFLICT** — recommend 0.0, documented as benchmark-specific |
| Judge max output tokens | none | none | 1024 (single path) | no cap | S4 `common.py`; S8 | **NOT SPECIFIED** |
| Scoring scale | 1–10 (explicit) | none | 1–10, `[[rating]]` | clamps to 1–10 | S1 p8; S4 `common.py`; S8 `mtbench.py:124-135` | **CONFIRMED** |
| Turn vs conversation scoring | per turn (Fig. 7) | none | per turn | per turn | S1 Fig. 7; S5 §5; S8 | **CONFIRMED** |
| Aggregation | single number, method unstated | none | mean over all 160 turn scores | `mean_score = mean(turn-1)` (turn 2 discarded) | S5 §5; S4 `show_result.py`; S8 `mtbench.py:357-359` | **CONFLICT** — implementation contradicts the benchmark definition |
| Parse failure | none | none | `rating = -1` | score 1, cached (`mtbench.py:127-135,183-186`) | S4 `common.py`; S8 | **NOT SPECIFIED** + defect (fabricated score) |
| Multiple judges / retries | none | none | one judgment per turn; API retried 8× | one call, no retry | S4 `common.py`; S8 | **NOT SPECIFIED** |

### 4.4 Verified / inferred / unresolved

* **Verified:** 80 questions, 2 turns, 160-score mean, GPT-4 single-answer, 1–10, per-category
  upstream temperatures, upstream 1024-token cap, judge temperature 0, `[[rating]]` parsing,
  `-1` on parse failure, per-turn output in FastChat and in the NBF paper's Fig. 7.
* **Inferred (not paper requirements):** that the authors used the full 80-question set; that they
  averaged both turns; that they used GPT-4 single-answer grading; that they used the benchmark's
  per-turn scoring. Each is the benchmark's definition and the most defensible default, but the
  paper states none of them.
* **Unresolved:** release, judge model, judge prompt body, judge temperature scope, reference-answer
  use, token cap, system prompt, aggregation choice by the authors, whether a subset was used.

---

## 5. XSTest methodology audit

### 5.1 What the NBF paper actually specifies

| Location | Statement |
|---|---|
| p8 §5.1 | "We further compare the **over-refusal rate (refused queries over all)** on XSTest Röttger et al. (2024) ..." |
| p8 Table 1 | GPT-4o: XSTest **0.004**; GPT-4o + steering: **0.026** |
| p9 Table 3 | llama 0.078 / 0.178 / 0.217 / 0.087 / 0.096; Phi-4 0.087 / 0.052 / 0.139 / 0.087 / 0.100 |
| p24 Table 10 | GPT-5 0.052 → 0.122; Claude Sonnet 4.5 0.035 → 0.074 |
| p26 Table 13 | llama original 0.078, system prompt 0.178, SFT **0.237**, DPO 0.226, KTO 0.100, steering η=1e−3 0.096; Phi-4 original **0.100**, SFT 0.139, ... |
| p22 B.1 | refusal substitution is stated **only** for MTBench; no XSTest-specific rule |

No XSTest version, subset, detector, judge, class taxonomy, partial-refusal policy, aggregation or
filtering rule is stated anywhere in the paper.

### 5.2 What the benchmark defines

* **Composition (abstract, Table 1, Table 5):** **250 safe prompts** across ten types (25 each) and
  **200 unsafe contrast prompts** across eight types (25 each); total 450.
* **Reported protocol (Table 2):** refusal rates are reported **separately** for
  "n=250 safe prompts" and "n=200 unsafe prompts" — never pooled. The benchmark's over-refusal
  target is the safe subset.
* **What counts as a refusal (Table 1 caption, §4.2):** full refusal **and partial refusal** are
  counted; they "add up to 100% with full compliance".
* **Detectors:** `classify_completions_strmatch.py` (lower-case prefix matching against a fixed
  list → only `2_full_refusal` / `1_full_compliance`; **no partial class**) and
  `classify_completions_gpt.py` (GPT-4, `temperature=0`, `max_tokens=16`, `top_p=1`,
  `frequency_penalty=0`, `presence_penalty=0`; three classes including `3_partial_refusal`).
* **Detector comparison (Table 2):** string-match collapses partial refusals and reports a single
  number; the GPT-4 column reports "full + partial".
* **Benchmark-internal slip:** §4.3 says "we evaluate models on the 200 safe prompts", while the
  abstract, Table 1, Table 2 and Table 5 all say 250 safe / 200 unsafe. Treat 250/200 as
  authoritative; the sentence is a typo.
* **No official XSTest score script** — the repository ships classifiers and a notebook
  (`analysis.ipynb`, 136 168 B) but no single "compute over-refusal" entry point.

### 5.3 Methodology table

| Parameter | Paper specification | Author implementation | Upstream implementation | Guardbound status | Evidence | Resolution |
|---|---|---|---|---|---|---|
| Dataset release | none | none | commit `475f10bf…` (no tags/releases); `xstest_prompts.csv` `ad38204d…`, 38 719 B | JSONL loader; `data/raw/xstest.jsonl` absent | S1 p8; S2; S6 tree API | **UNVERIFIED** — paper release unknown |
| Prompt count | none | none | 450 (250 safe + 200 unsafe) | whatever the file contains | S7 abstract, Tables 1/5 | **PARTIALLY CONFIRMED** |
| Prompt types | none | none | 10 safe types ×25; 8 unsafe types ×25 | `type` read into `label`, never used | S7 Tables 1/5; S6 CSV schema | **PARTIALLY CONFIRMED** |
| Safe/unsafe labels | none | none | `type` starting `contrast_` = unsafe (README); CSV also carries `label` ∈ {safe, unsafe} | label stored per example, no filtering | S6 README; S8 `over_refusal.py:74` | **PARTIALLY CONFIRMED** |
| Denominator | "refused queries over all" (ambiguous) | none | 250 safe (primary) and 200 unsafe (separate) | all rows in the supplied file | S1 p8; S7 Table 2; S8 `over_refusal.py:177-178` | **NOT SPECIFIED** — recommend 250 safe, unsafe reported separately; needs authorization |
| Metric | refusal rate | none | refusals / denominator | `is_refusal(response)` count / rows | S1 p8; S7 §4.2; S8 | **PARTIALLY CONFIRMED** |
| Safe-subset filtering | none | none | yes, by construction | none | S7 §4.3; S8 | **NOT SPECIFIED** |
| Unsafe prompts | none | none | separate safety metric | folded into the same rate | S7 Table 2; S8 | **NOT SPECIFIED** |
| Detector | none | none | strmatch (2-class) or GPT-4 (3-class) | third detector: 13-pattern regex (`refusal_detector.py:23-37`) | S1; S6 scripts; S8 | **NOT SPECIFIED** — no detector is paper-selected |
| Detector classes | none | none | 2 vs 3 | binary | S6 scripts | **NOT SPECIFIED** |
| Partial refusals | none | none | counted as refusals by manual/GPT protocol; absent from strmatch | regex may flag a refusal prefix inside a mixed answer | S7 §4.2, Table 2; S8 | **NOT SPECIFIED** |
| Disclaimers / alternatives | none | none | GPT prompt: "rejects for not meaningful answer" = compliance; "only offers alternatives" = full refusal | generic "not appropriate / not possible" patterns may flag | S6 `classify_completions_gpt.py`; S8 | **NOT SPECIFIED** |
| Detector model/config | none | none | GPT-4, T=0, 16 tokens, top_p=1 | no judge (regex) | S6 | **NOT SPECIFIED** |
| Malformed classifier output | none | none | upstream stores the raw completion text without parsing/validation | n/a | S6 `classify_completions_gpt.py` | **NOT SPECIFIED** |
| Aggregation | none | none | per-type + total (equal n per type, so micro = macro) | one overall rate | S7 Table 1; S8 | **NOT SPECIFIED** |
| Refusal substitution | B.1 states it for MTBench only | none | n/a | none on this path | S1 p22; S8 | **NOT SPECIFIED** for XSTest |
| Steered semantics | filter is a runtime rule; how a blocked *safe* prompt is scored is unstated | none | n/a | CLI loads the barrier and never applies it (`eval_over_refusal.py:96-107`); a filtered steered turn returns `response=None` (`steered_chat.py:203-218`) which `is_refusal` treats as **not refused** | S1 §5.2; S8 | **NOT SPECIFIED** + implementation defect |
| Target temperature | "all temperatures 0.7" | none | n/a (benchmark has no fixed temp) | hard-coded 0.0 (`over_refusal.py:162`) | S1 p8; S8 | **CONFLICT** |
| XSTest values reproducibility | two different values for the same condition (Table 3 vs Table 13) | none | n/a | n/a | S1 Tables 3 & 13 | **CONFLICT** (paper-internal) — exact reconstruction impossible |

### 5.4 Verified / inferred / unresolved

* **Verified:** 250 safe (10×25) + 200 unsafe (8×25); refusal rates reported separately per subset;
  full + partial refusals both count in the benchmark's manual/GPT protocol; the two upstream
  detectors and their exact behaviour; the CSV columns (`id, prompt, type, label, focus, note`) and
  the `contrast_` rule.
* **Inferred (not paper requirements):** that the NBF paper's "over-refusal" is the safe-prompt
  rate on 250 prompts (the benchmark's definition, and the only reading under which the metric is
  "over-refusal"); that partial refusals count (benchmark protocol).
* **Unresolved:** release, denominator (450 vs 250), detector, partial-refusal policy, whether the
  paper filtered or normalised responses, how a Q-filter rejection is scored.
* **Arithmetic check (inference, not proof):** none of Table 1's XSTest values is an integer
  multiple of 1/250 (0.078 = 19.5/250; 0.026 = 6.5/250) or of 1/450 (0.078 = 35.1/450;
  0.026 = 11.7/450). The reported values therefore cannot come from one deterministic pass over
  either denominator; they are consistent with averaging over several samples/runs or with a
  different tabulation. This reinforces **UNVERIFIED** for the paper's exact procedure.

---

## 6. Discrepancy matrix

### 6.1 Paper vs author vs upstream vs Guardbound

| # | Discrepancy | Paper | Author code | Upstream | Guardbound | Evidence | Proposed resolution | Authorization? |
|---|---|---|---|---|---|---|---|---|
| D-01 | MT-Bench release | unstated | absent | `v0.2.36` / `b494d0c6…` | none pinned in config | S1, S2, S4 | adopt the pin as reference; record as "paper release unverified" | yes (acquisition) |
| D-02 | MT-Bench generation temperature | "all 0.7" | absent | per-category 0/0.1/0.7 | 0.7 | S1 p8, S4 `common.py` | paper wins (hierarchy #1): 0.7, via `--force-temperature 0.7` | yes (acknowledge deviation) |
| D-03 | MT-Bench judge model | unstated | absent | GPT-4 | GPT-4o default | S5, S4, S8 | use GPT-4 for benchmark fidelity; keep GPT-4o only if consistency with the ASR judge is preferred | yes |
| D-04 | MT-Bench judge prompt | unstated | absent | Figure 6 / `single-v1-multi-turn` | local invented prompt; CLI flag ignored | S5, S4, S8 | adopt the upstream multi-turn template for turn 2 and the Figure-6 template for turn 1; fix the CLI branch | yes |
| D-05 | MT-Bench aggregation | single value, method unstated | absent | mean of 160 turn scores | mean of turn-1 only | S5 §5, S4 `show_result.py`, S8:357-359 | mean of both turns (160) | no (defect) |
| D-06 | MT-Bench turn-2 question | unstated | absent | `turns[1]` | `turn2_question` or **`reference`** or invented prompt | S4, S8:302 | require `turns[1]`; fail loudly if absent | no (defect) |
| D-07 | MT-Bench token cap / system prompt | unstated | absent | 1024 / model template | none / none | S4, S8 | set 1024 and record the template choice | yes |
| D-08 | Judge decoding | "all 0.7" (scope unclear) | absent | T=0, 1024 tokens | T=0, no cap | S1, S4, S8 | keep T=0 (benchmark) and document the paper-scope ambiguity; cap 1024 | yes |
| D-09 | Judge failure handling | unstated | absent | `-1` | score **1**, cached | S4, S8:127-135 | propagate a typed failure; never fabricate a score | no (defect) |
| D-10 | XSTest release | unstated | absent | `475f10bf…` | none pinned in config | S1, S2, S6 | adopt the pin; record "paper release unverified" | yes (acquisition) |
| D-11 | XSTest denominator | "refused queries over all" | absent | 250 safe (+200 unsafe separately) | all rows | S1, S7, S8 | 250 safe primary; unsafe as a separate diagnostic | yes |
| D-12 | XSTest detector | unstated | absent | strmatch / GPT-4 3-class | regex heuristic | S1, S6, S8 | GPT-4 3-class with partial refusals counted; report strmatch as a secondary sensitivity check | yes |
| D-13 | XSTest steered semantics | unstated | absent | n/a | barrier loaded, never applied | S8 `eval_over_refusal.py` | wire the Q-filter; count a rejected safe prompt as a refusal (recommended) | yes |
| D-14 | Utility target temperature | "all 0.7" | absent | n/a | XSTest 0.0 | S1, S8 | 0.7 per the paper, documented as a deviation from the benchmark's deterministic style | yes |
| D-15 | Refusal string codepoint | U+2019 ("I’m") | absent | n/a | U+0027 ("I'm") | S1 p22, S8 | keep ASCII (semantically identical) and record the difference | yes (cosmetic) |

### 6.2 Paper-internal and benchmark-internal inconsistencies

| # | Inconsistency | Values | Consequence |
|---|---|---|---|
| P-01 | MT-Bench substitution trigger | B.1 "positive values" (h > 0) vs §5.2 filter `h + η ≥ 0` | at η = 1e-3 the two differ on h ∈ [−1e-3, 0); B.1 is the benchmark-specific statement, so Guardbound's h > 0 is defensible but must be labelled |
| P-02 | XSTest value for llama-3-8b + LoRA SFT | Table 3 **0.217** vs Table 13 **0.237** | the same condition has two values; the paper's XSTest column cannot be reproduced exactly |
| P-03 | XSTest value for Phi-4 original | Table 3 **0.087** vs Table 13 **0.100** | idem; also suspiciously equals Table 3's steering η=1e−3 cell |
| P-04 | XSTest §4.3 wording | "the 200 safe prompts" vs 250 safe everywhere else | typo; use 250 |
| P-05 | Phase 30 record of MT-Bench judge max tokens | 2048 vs upstream 1024 (single path) | corrected in Phase 31 and here |
| P-06 | Phase 31 XSTest readme blob | `2850b164…` (2 772 B, `main`) vs `92bfa9c9…` (2 535 B, pinned commit) | provenance field in `readiness.py` must be corrected in a future authorized change |

---

## 7. Guardbound utility implementation āaudit

Status vocabulary: **IMPLEMENTED CORRECTLY** · **IMPLEMENTED WITH DEVIATION** · **MISSING** ·
**BLOCKED BY DATA** · **BLOCKED BY METHODOLOGY** · **NOT VERIFIED**.

### 7.1 MT-Benchā

| #   | Item                                      | Status                                                                                                                           | Evidence                                              |
| --- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 1   | Two-turn execution with history           | IMPLEMENTED CORRECTLY                                                                                                            | `mtbench.py:292-311`                                  |
| 2   | Per-turn judging (turn 1 and turn 2)      | IMPLEMENTED CORRECTLY                                                                                                            | `mtbench.py:334-343`                                  |
| 3   | 1–10 scale parsing                        | IMPLEMENTED CORRECTLY (clamped)                                                                                                  | `mtbench.py:124-135`                                  |
| 4   | Aggregate score                           | **IMPLEMENTED WITH DEVIATION (defect)** — `mean_score` is the turn-1 mean; turn 2 is discarded                                   | `mtbench.py:357-359`                                  |
| 5   | Turn-2 question sourcing                  | **MISSING / defect** — falls back to `reference` (an *answer*) then to an invented prompt                                        | `mtbench.py:302`                                      |
| 6   | Judge prompt                              | **MISSING** — local invented prompt; `--judge-prompt` never reaches `judge_single_turn` (unreachable branch)                     | `mtbench.py:106-112,201-205`; `eval_mtbench.py:25-26` |
| 7   | Two-turn judge prompt                     | **MISSING in the driver** — `DEFAULT_MTBENCH_JUDGE_PROMPT` is defined but never used by `evaluate_mtbench`                       | `mtbench.py:91-103`                                   |
| 8   | Judge model                               | IMPLEMENTED WITH DEVIATION — GPT-4o default vs benchmark GPT-4                                                                   | `eval_mtbench.py:23`                                  |
| 9   | Judge temperature                         | IMPLEMENTED WITH DEVIATION — 0.0 vs the paper's unqualified "all 0.7"                                                            | `mtbench.py:181,225`                                  |
| 10  | Judge failure handling                    | **IMPLEMENTED WITH DEVIATION (defect)** — an exception yields `raw="1"` → score 1, then **cached**                               | `mtbench.py:180-187,224-231`                          |
| 11  | Reference-guided judging                  | MISSING                                                                                                                          | no reference use anywhere                             |
| 12  | Max tokens / stop strings / system prompt | MISSING — no control on any of them                                                                                              | `mtbench.py:294`; `openai_client.py`                  |
| 13  | NBF refusal substitution                  | IMPLEMENTED CORRECTLY for B.1 (h > 0, turn 2)                                                                                    | āā`mtbench.py:313-327`; `post_processing.py:51-65ā`āā |
| 14  | Substitution string                       | IMPLEMENTED WITH DEVIATION — U+0027 vs paper's U+2019; 209 chars both                                                            | `post_processing.py:18-23`                            |
| 15  | Per-example and sidecar reporting         | IMPLEMENTED CORRECTLY — emits `MTBench`, `MTBench_turn1`, `MTBench_turn2` rows with model/defense/eta/judge/n                    | `mtbench.py:374-405`                                  |
| 16  | Dataset version in results                | MISSING — `EvaluationResult.dataset` is the literal `"MTBench"`; no commit/blob/sha                                              | `reporting.py:29-51`; `mtbench.py:381`                |
| 17  | Dataset loader                            | BLOCKED BY DATA + schema mismatch: requires `question`; upstream provides `turns[]`/`question_id`; no transformation implemented | `mtbench.py:51-84`; `readiness.py`                    |
| 18  | Steered metrics for both arms             | IMPLEMENTED CORRECTLY for MTBench (barrier applied when supplied)                                                                | `mtbench.py:281-327`                                  |

### 7.2 XSTest / over-refusal

| # | Item | Status | Evidence |
|---|---|---|---|
| 19 | Prompt execution and refusal counting | IMPLEMENTED CORRECTLY in isolation | `over_refusal.py:155-178` |
| 20 | Denominator | **BLOCKED BY METHODOLOGY** — always all rows; no safe-subset option, no per-label split | `over_refusal.py:177-178` |
| 21 | Label handling | IMPLEMENTED WITH DEVIATION — `label`/`type` read but never used for filtering or stratification | `over_refusal.py:74` |
| 22 | Refusal detector | IMPLEMENTED WITH DEVIATION — a project-specific 13-pattern regex, neither upstream detector; empty response → not a refusal | `refusal_detector.py:23-54` |
| 23 | Detector classes / partial refusals | MISSING — binary only | `refusal_detector.py` |
| 24 | Steered mode | **MISSING (defect)** — the CLI loads the barrier and never applies it; `defense` is only a label | `eval_over_refusal.py:96-107` |
| 25 | Filtered-query accounting | MISSING — `SteeredLLMChat.chat` returns `response=None` when filtered; `is_refusal(None)` is False, so a blocked safe prompt would be counted as *not* refused | `steered_chat.py:203-218`; `refusal_detector.py:52-54` |
| 26 | Temperature | IMPLEMENTED WITH DEVIATION — hard-coded 0.0 vs paper 0.7 | `over_refusal.py:160-163` |
| 27 | Aggregation | MISSING — single overall rate; no per-type or macro/micro | `over_refusal.py:177-203` |
| 28 | Dataset version in results | MISSING — dataset name only | `over_refusal.py:196-201` |
| 29 | CSV→JSONL transformation | MISSING — loader expects JSONL | `over_refusal.py:51-85`; `readiness.py` |
| 30 | Readiness schema | IMPLEMENTED WITH DEVIATION — accepts `prompt|text`, optional `type`; does not require/validate `label`, does not verify the 450/250/200 split | `readiness.py` |

### 7.3 Cross-cutting

| # | Item | Status | Evidence |
|---|---|---|---|
| 31 | Two CLIs with distinct paths | IMPLEMENTED WITH DEVIATION — `run_full_evaluation.py` reads `data/raw/mtbench/test.jsonl` and `data/raw/xstest/test.jsonl`, while the Phase-7 CLIs read `data/raw/mtbench.jsonl` / `xstest.jsonl` | `run_full_evaluation.py:319,350`; `eval_mtbench.py:82` |
| 32 | Duplicated over-refusal loop | IMPLEMENTED WITH DEVIATION — `run_full_evaluation.py:356-366` reimplements counting instead of calling `evaluate_over_refusal` | `run_full_evaluation.py:338-397` |
| 33 | Both utility metrics through one registry | IMPLEMENTED CORRECTLY — both emit `EvaluationResult` rows consumable by Phase 9 | `evaluation/__init__.py:49-65` |
| 34 | Fail-closed blocking | IMPLEMENTED CORRECTLY — both CLIs exit 2 with `BLOCKED — DATA REQUIRED`, nothing fabricated | `eval_mtbench.py:82-92`; `eval_over_refusal.py:83-90` |
| 35 | Utility status in manifests | IMPLEMENTED CORRECTLY — `utility_manifest_block` recomputes from the filesystem, `table1_ready=false` | `_cloud_common.py:909-975`; Phase-31 tests |
| 36 | End-to-end utility evaluation | BLOCKED BY DATA + BLOCKED BY METHODOLOGY + NEEDS CHANGES | — |

**Conclusion:** the utilities are scaffolds, not a working evaluation. Four defects would corrupt a
real run even after the data arrives: MTBench aggregation (4), turn-2 sourcing (5), judge-prompt
plumbing + fabricated failure score (6, 10), and the unwired XSTest filter (24).

---

## 8. Dataset acquisition and integrity plan

Nothing below is authorised for execution. This is the specification to run **after** the
decisions in §13.

### 8.1 MT-Bench

| # | Item | Specification |
|---|---|---|
| 1 | Official repository | `https://github.com/lm-sys/FastChat` |
| 2 | Revision | tag `v0.2.36` (commit `b21d0f78…`) **and** dataset commit `b494d0c6…`; the dataset blobs are identical at both |
| 3 | Exact paths | `fastchat/llm_judge/data/mt_bench/question.jsonl`; `fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl`; (harness) `fastchat/llm_judge/data/judge_prompts.jsonl` |
| 4 | Git blob SHA-1 | `464e2c22f7a021a0f9a584d232338a9a17c40066`; `8b66387a3129cbb117178fe515491df553dd52df`; `d1e499f12d18c57338d883d43f5735e512c00d8c` |
| 5 | Size | 48 929 B; 51 109 B; 10 457 B (verified from tree metadata) |
| 6 | SHA-256 status | **not computed** — requires the bytes; compute on first authorized acquisition and record |
| 7 | License | **not verified in this phase**; inspect the repository `LICENSE` before acquisition |
| 8 | Required files | `question.jsonl` (always); `reference_answer/gpt-4.jsonl` **iff** reference-guided judging is chosen for math/reasoning/coding; `judge_prompts.jsonl` **iff** verbatim judge-prompt reproduction is required |
| 9 | Unnecessary | `model_answer/`, `model_judgment/`, `vicuna_bench/`, `misc/`, the pre-generated Google-Drive bundle |
| 10 | Proposed local layout | `data/utility/mtbench/questions/question.jsonl`, `.../reference_answers/gpt-4.jsonl`, `.../provenance.json`, `.../checksums.sha256` |
| 11 | Integrity procedure | recompute the git blob SHA-1 as `sha1(b"blob " + len(bytes) + b"\0" + bytes)` and compare; then record SHA-256; abort on any mismatch; assert 80 rows, each with `question_id`, `category`, two `turns` |
| 12 | Provenance metadata | repository URL, tag, tag commit, dataset commit, per-file blob SHA-1, byte size, SHA-256, retrieval timestamp, retrieval tool, licence, paper-version status |
| 13 | Paper version | **uncertain** — record `PAPER VERSION UNVERIFIED` |

### 8.2 XSTest

| # | Item | Specification |
|---|---|---|
| 1 | Official repository | `https://github.com/paul-rottger/xstest` |
| 2 | Revision | commit `475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5` — **no tags or releases exist**, so the commit is the only immutable pin |
| 3 | Exact paths | `xstest_prompts.csv`; (docs) `readme.md`; (detector code) `evaluation/classify_completions_strmatch.py`, `evaluation/classify_completions_gpt.py` |
| 4 | Git blob SHA-1 | `ad38204d70b3964387f9690169d11afbc4756d01`; `92bfa9c944ad3c5bd649d00bde2c33e90fcd238a`; `aa8fa43a2043c7d400923bd4ffb628154aca1b7c`; `b9f911978a4ee5c91aa65239a59c2b979cd61576` |
| 5 | Size | 38 719 B; 2 535 B; 1 368 B; 2 721 B |
| 6 | SHA-256 status | **not computed** (same reason) |
| 7 | License | **CC-BY-4.0** (verified from repository metadata); attribution required |
| 8 | Required files | `xstest_prompts.csv`; the chosen classifier script; (if GPT detector) its prompt text |
| 9 | Unnecessary | `model_completions/*` (5 files, 0.3–0.5 MB each), `evaluation/automated_evaluation_labels/*` (10 files), `evaluation/analysis.ipynb` |
| 10 | Proposed local layout | `data/utility/xstest/prompts/xstest_prompts.csv`, `.../prompts/xstest_prompts.jsonl` (transformed), `.../provenance.json`, `.../checksums.sha256` |
| 11 | Integrity procedure | verify the git blob SHA-1; record SHA-256; assert 450 rows; assert columns `id, prompt, type, label, focus, note`; assert 250 `label == safe` and 200 `label == unsafe`; assert `type` starting `contrast_` ⇔ `label == unsafe`; assert the prefix and label rules agree before use |
| 12 | Provenance metadata | as MTBench, plus: subset selected (250 safe / 450), detector id + version + prompt hash, label-rule cross-check result |
| 13 | Paper version | **uncertain** — `PAPER VERSION UNVERIFIED` |

### 8.3 Proposed directory structure (proposal only — do not create)

```text
data/
  utility/
    mtbench/
      questions/question.jsonl
      reference_answers/gpt-4.jsonl
      provenance.json
      checksums.sha256
    xstest/
      prompts/xstest_prompts.csv
      prompts/xstest_prompts.jsonl
      provenance.json
      checksums.sha256
```

### 8.4 Integrity rules to carry into the acquisition step

* **Git blob SHA-1 ≠ file SHA-256.** The blob hash covers `"blob <len>\0" + content`; record both,
  never conflate them, and never invent a SHA-256.
* No local-integrity claim may be made before the file exists; today both datasets are
  metadata-verified only.
* Do not download data merely to compute a hash; do not fetch bodies through metadata endpoints.
* Validate structure (row count, columns, label counts) **before** any evaluation run; a mismatch
  must abort, not warn.

---

## 9. Frozen artifact verification

All values below were recomputed in this phase from the local files.

| Artifact | SHA-256 | Status |
|---|---|---|
| HarmBench tasks (`NBF-LLM/data/test/harmbench_tasks.json`, 38 332 B) | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` | **VERIFIED** — matches the Phase 30/31 value |
| NBF checkpoint (`models_best_nbf_released.pth`, 11 757 138 B) | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` | **VERIFIED** |
| Reference `generate.py` (5 607 B) | `8b2dcd8520def27f62a5f4a96766d9ef07b3c462d6898adb3b0ce067e07dcb29` | **VERIFIED** — matches `8b2dcd…` |
| `_official_prompts.py` (12 866 B) | `b3cd2d074506429737ca34db91e4b4ee4637cec3d81c67e22c8c0129294a92e2` | **VERIFIED** — matches `b3cd2d…` |
| `src/guardbound/evaluation/mtbench.py` (14 701 B) | `4b8b2d9b70e397495ca8459ba355b380d9a203bf8abe4486cf76671e9a6691b5` | tracked and unmodified (`git status` clean); recorded for change detection |
| `src/guardbound/evaluation/over_refusal.py` (6 731 B) | `83d7fb14946127ea6c1fe0928370ce964f1de073a7fa997826b040b966302b61` | tracked and unmodified; recorded for change detection |

Also confirmed: no change to attack-ASR script contents (`git status` shows only the pre-existing
Phase 22–31 modifications, none from this phase), no change to the NBF implementation, both
evaluation arms and `configs/paper_reproduction_r1_gpt4o.yaml` untouched, and no dataset file
persisted (`data/raw` holds only the pre-existing HarmBench/Circuit-Breakers files; neither
`data/raw/mtbench.jsonl`, `data/raw/xstest.jsonl`, nor any `data/raw/mtbench`/`data/raw/xstest`
directory exists).

Note on the reference file size: 5 607 **bytes** vs the 5 455 characters reported in Phase 31 —
the file contains non-ASCII UTF-8 characters (e.g. typographic quotes), so both figures describe
the same unchanged bytes.

---

## 10. Tests and validation results

Only five filesystem/provenance tests were run — no test that touches data, models or a network:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -m pytest `
  "tests/test_phase31_retry_semantics.py::test_utility_block_reports_pins_and_stays_blocked" `
  "tests/test_phase31_retry_semantics.py::test_manifest_records_upstream_dataset_pins" `
  "tests/test_phase30_retry_provenance.py::test_utility_block_is_blocked_and_lists_gaps" `
  "tests/test_phase30_retry_provenance.py::test_a_declaration_cannot_override_missing_data" `
  "tests/test_phase30_retry_provenance.py::test_utility_ready_with_synthetic_fixtures" `
  -q --no-header -p no:cacheprovider
```

| Metric | Result |
|---|---:|
| Collected | 5 |
| Passed | **5** |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Duration | 0.14 s |

No network request, model load, model inference, judge inference, API call or dataset read
occurred. The full suite was **not** re-run: nothing changed in this phase, and Phase 31 recorded
1277 collected / 1273 passed / 4 failed with four known environmental failures.

---

## 11. Prohibited-operation compliance statement

* **No dataset was downloaded, cloned, materialised, copied or modified.** No dataset body was
  retrieved in Phase 32. Dataset-path inspection used only the GitHub **git-trees** API (paths,
  blob SHA-1, byte size). The GitHub **contents** API — which inlines file bodies — was **not**
  used for any dataset path.
* Disclosed: Phase 31 had two contents-API metadata calls that also returned file bodies inline;
  those bytes were never persisted. Phase 32's plan (§8) explicitly forbids that endpoint.
* Disclosed: two **benchmark publications** (XSTest NAACL PDF, MT-Bench arXiv PDF) were downloaded
  to the session scratchpad and text-extracted. They are papers, not datasets; they were not
  copied into the repository, and the scratchpad is session-scoped.
* No MT-Bench or XSTest evaluation, no model inference, no judge inference, no paid API call, no
  credential read, no qualification, smoke test, pilot or experiment.
* No script, config, test, attack, NBF component, checkpoint or historical artifact was modified.
  `git status` for the utility modules is clean; the only new file is this report.
* No directory from §8.3 was created.

---

## 12. Remaining blockers

| # | Blocker | Type |
|---|---|---|
| 1 | Both datasets absent | DATA |
| 2 | Paper's benchmark releases unknown (both) | PROVENANCE |
| 3 | XSTest denominator (250 safe vs 450) unresolved in the paper | METHODOLOGY |
| 4 | XSTest detector unresolved in the paper | METHODOLOGY |
| 5 | XSTest steered/filtered-query accounting undefined | METHODOLOGY |
| 6 | MT-Bench judge model/prompt/decoding unresolved in the paper | METHODOLOGY |
| 7 | MT-Bench aggregation contradicts the benchmark in code | IMPLEMENTATION |
| 8 | MT-Bench turn-2 sourcing and judge-prompt plumbing broken | IMPLEMENTATION |
| 9 | Judge failure fabricates score 1 | IMPLEMENTATION |
| 10 | XSTest steered run does not apply the filter | IMPLEMENTATION |
| 11 | Utility temperature (0.0 on XSTest) conflicts with the paper | CONFLICT |
| 12 | Paper-internal XSTest value inconsistency prevents exact reproduction | PAPER |

---

## 13. Explicit decisions requiring user authorization

Each item states the recommended option first. Nothing below is executed by this phase.

| # | Decision | Options | Recommendation | Impact if unresolved |
|---|---|---|---|---|
| A1 | Acquire MT-Bench at the pinned revision | `v0.2.36` / different release / none | `v0.2.36` + `b494d0c6…`, record `PAPER VERSION UNVERIFIED` | MTBench column impossible |
| A2 | Acquire XSTest at the pinned revision | `475f10bf…` / different release / none | `475f10bf…`, record `PAPER VERSION UNVERIFIED` | XSTest column impossible |
| A3 | MT-Bench generation temperature | paper 0.7 / upstream per-category | **0.7** (paper explicit; obtainable via `--force-temperature 0.7`), documented as an upstream deviation | either choice must be declared |
| A4 | MT-Bench judge model | GPT-4 / GPT-4o | **GPT-4** for benchmark fidelity; GPT-4o only for consistency with the ASR judge | judge choice changes scores |
| A5 | MT-Bench judge prompt | Figure 6 (single-turn) / FastChat `single-v1-multi-turn` / both | **Figure 6 for turn 1 + upstream multi-turn for turn 2**; requires retrieving `judge_prompts.jsonl` (harness file, not dataset) | scores not comparable |
| A6 | MT-Bench judge temperature | 0 / 0.7 | **0** (benchmark), document the paper's ambiguous global 0.7 | non-deterministic judging |
| A7 | Reference-guided judging | on for math/reasoning/coding / off | **on** (upstream) if `reference_answer/gpt-4.jsonl` is acquired | deviates from benchmark |
| A8 | MT-Bench token cap / system prompt | 1024 + model template / unbounded | **1024 + model template** | truncated/over-long answers |
| A9 | MT-Bench aggregation | mean of 160 turn scores / turn-1 only / per-turn reported only | **mean of 160**, with turn-1 and turn-2 reported alongside | wrong headline number |
| A10 | XSTest denominator | 250 safe / 450 / both | **250 safe primary**, 200 unsafe as a separate diagnostic | metric meaning changes |
| A11 | XSTest detector | GPT-4 3-class / strmatch / regex | **GPT-4 3-class**, partial refusals counted as refusals; report strmatch as a sensitivity check | metric not comparable |
| A12 | Q-filter rejection accounting on XSTest | counts as refusal / excluded / separate | **counts as refusal** (a blocked safe prompt is a refusal), and record the filtered count | over-refusal understated |
| A13 | Utility target temperature for XSTest | paper 0.7 / deterministic 0.0 | **0.7** (paper explicit), documented | conflicts with benchmark style |
| A14 | Refusal-string codepoint | ASCII `'` / typographic `’` | **ASCII** (semantically identical), recorded | cosmetic |

---

## 14. Final readiness classification

| Component | Classification | Basis |
|---|---|---|
| **MT-Bench methodology** | **`PARTIALLY RESOLVED`** | Every protocol element is established at the benchmark level and cross-checked against FastChat; the paper's own choices (release, judge, prompt, aggregation reference, temperature scope) remain unstated, and one paper-vs-upstream conflict (temperature) needs a decision |
| **XSTest methodology** | **`PARTIALLY RESOLVED`** | Composition, label rule and both detectors are established; denominator, detector choice, partial-refusal policy and filtered-query accounting are not specified by the paper |
| **MT-Bench dataset provenance** | **`PARTIALLY VERIFIED`** | Repository, tag, commit and per-file blob SHA-1 pinned and byte sizes verified from metadata; SHA-256 unavailable without acquiring the data; the authors' actual release is unverified |
| **XSTest dataset provenance** | **`PARTIALLY VERIFIED`** | Commit and per-file blob SHA-1 pinned (no tags exist); byte sizes verified; SHA-256 unavailable; the authors' actual release is unverified |
| **Dataset acquisition plan** | **`READY`** | Exact URLs, revisions, paths, blob hashes, sizes, licences, required/unnecessary files, local layout, verification procedure and provenance fields are specified for both (execution still requires authorization) |
| **Utility implementation** | **`NEEDS CHANGES`** | Scaffolding exists and fails closed correctly, but the MTBench aggregate discards turn 2, turn-2 sourcing is wrong, `--judge-prompt` is inert, judge failure fabricates score 1, the XSTest run never applies the filter, and the XSTest temperature is 0.0 |
| **Full Table 1** | **`NOT READY`** | Neither utility column can be produced: data absent, methodology partially resolved, and the implementation needs the changes above |

**Table 1 remains NOT READY.** No classification above authorises a dataset download, a directory
creation, a code change, an evaluation run or any paid call. Phase 32 ends here.
