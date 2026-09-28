# Ornith-1.0-9B-GGUF — qualification result

**NOT QUALIFIED — MODEL_INTERFACE_INCOMPATIBLE.**

The model could not be loaded by the frozen attacker interface. Evidence:

- `stage0_load_gate_result.json` — the empirical load attempt (3.65 s, 0.000 GB downloaded), classified
  `MODEL_LOAD_FAILURE` / `ModelLoadError`:
  the tokenizer could not be instantiated.
- No `config.json` in the repository (`AutoConfig` raises "Unrecognized model").
- Local build reports `transformers.utils.import_utils.is_gguf_available() == False`.
- The repo contains five GGUF quant files (Q4_K_M … bf16); the frozen loader passes only a
  model id, and selecting a quant requires a `gguf_file=` argument — i.e. a loader code change.

No workaround was implemented. See `PHASE_15_ATTACKER_QUALIFICATION_REPORT.md` at the repo root.
