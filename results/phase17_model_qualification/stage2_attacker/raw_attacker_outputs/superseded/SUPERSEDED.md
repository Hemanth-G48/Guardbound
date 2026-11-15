# SUPERSEDED raw outputs — Phase 17 Stage 2, attacker JSON qualification

`attacker_json_Qwen__Qwen3.8-27B_run1_harness_defect.jsonl` is the raw output of
qualification RUN 1 for `Qwen/Qwen3.8-27B` and is **SUPERSEDED** by the records in
`../attacker_json_Qwen__Qwen3.8-27B.jsonl` (RUN 2).

Why RUN 1 was invalidated: the qualification script kept a strong reference to the
loaded model (`model = client._pipeline.model`) and released it while that reference
was still alive. The release therefore freed nothing, and the two reloads that
followed failed for memory reasons. RUN 1 reported 18/24 valid JSON with 6 steps
raising `ValueError: Could not load model ... with any of the following classes`.
Those six failures are artifacts of the measurement, not behaviour of the model.

RUN 2 (the corrected harness: references dropped before release) reported 23/24
valid JSON with a single, separate failure (a short prose reply, classified as
`AttackGenerationError`). The corrected values are the ones carried in
`candidate_qwen38_feasibility.json`.

The superseded evidence is kept rather than deleted, per the project convention of
marking superseded artifacts instead of overwriting them.
