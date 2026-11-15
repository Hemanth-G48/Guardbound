"""Phase 14.7 — derive the frozen-stack config from the existing three-model config.

Declarative only: this PATCHES MODEL IDS in a copy. It does not touch any
production code, prompt, schema, decoder or algorithm. The source config is read
and left untouched; a new config file is written for this validation gate.

Frozen stack (Phase 14.5/14.6 qualified):
    attacker  Qwen/Qwen3-4B-Instruct-2507
    target    microsoft/Phi-4-mini-instruct
    evaluator meta-llama/Llama-3.2-3B-Instruct
"""
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
SRC = ROOT / "configs" / "reproduction_three_model.yaml"
DST = ROOT / "configs" / "reproduction_phase14_frozen.yaml"

ATTACKER = "Qwen/Qwen3-4B-Instruct-2507"
TARGET = "microsoft/Phi-4-mini-instruct"
EVALUATOR = "meta-llama/Llama-3.2-3B-Instruct"

cfg = yaml.safe_load(SRC.read_text(encoding="utf-8"))
orig_ids = {r: cfg["models"][r]["model"] for r in ("attacker", "target", "evaluator")}
print("source config:", SRC.name)
print("  before:", orig_ids)

# --- patch model ids ------------------------------------------------------- #
cfg["models"]["attacker"]["model"] = ATTACKER
cfg["models"]["target"]["model"] = TARGET
cfg["models"]["evaluator"]["model"] = EVALUATOR

# flattened views kept in sync (consumed by sibling scripts / validation)
if "attacker" in cfg:
    cfg["attacker"]["model"] = ATTACKER
if "target" in cfg:
    cfg["target"]["model"] = TARGET
if "evaluator" in cfg:
    for k in ("rubric_model", "refusal_model", "disclaimer_model",
              "actor_step_judge_model"):
        if k in cfg["evaluator"]:
            cfg["evaluator"][k] = EVALUATOR
if "evaluation" in cfg and "judge_model" in cfg["evaluation"]:
    cfg["evaluation"]["judge_model"] = EVALUATOR

cfg["experiment"]["name"] = "phase14_7_18_run_production_validation"
cfg["experiment"]["output_dir"] = "results/phase14_7"

cfg["_phase14_7_provenance"] = {
    "derived_from": SRC.name,
    "purpose": "Phase 14.7 18-run production validation gate, frozen qualified stack",
    "model_ids_before": orig_ids,
    "model_ids_after": {"attacker": ATTACKER, "target": TARGET, "evaluator": EVALUATOR},
    "note": "Only model ids and experiment metadata were patched. NBF checkpoint, "
            "threshold, dataset, attacks, max_turns and generation parameters are "
            "inherited unchanged from the source config.",
}

DST.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("wrote:", DST)

after = {r: cfg["models"][r]["model"] for r in ("attacker", "target", "evaluator")}
print("  after :", after)
print("  distinct:", len(set(after.values())) == 3)

print("\n--- inherited frozen values (unchanged) ---")
print("  nbf.checkpoint      :", cfg["nbf"]["checkpoint"])
print("  nbf checkpoint sha  :", cfg["nbf"]["checkpoint_sha256_expected"])
print("  nbf.threshold       :", cfg["nbf"]["threshold"])
print("  nbf.state_dimension :", cfg["nbf"]["state_dimension"])
print("  nbf.predictor_classes:", cfg["nbf"]["predictor_classes"])
print("  nbf.initial_state   :", cfg["nbf"]["initial_state"])
print("  attacks.max_turns   :", cfg["attacks"]["max_turns"])
print("  attacks.enabled     :", cfg["attacks"]["enabled"])
print("  dataset.path        :", cfg["dataset"]["path"])
print("  experiment.seed     :", cfg["experiment"]["seed"])
print("  embedding.model     :", cfg["embedding"]["model"])

sha = hashlib.sha256(DST.read_bytes()).hexdigest()
print(f"\n  new config sha256   : {sha[:16]}")
print("\nOK - source config untouched:",
      SRC.read_text(encoding='utf-8').find("Qwen/Qwen3.5-4B") != -1)
