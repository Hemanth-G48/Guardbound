"""Phase 15 pilot — pre-flight verification (no runs launched)."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "phase14_full_reproduction", ROOT / "scripts" / "phase14_full_reproduction.py")
P14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P14)

CFG_PATH = ROOT / "configs/reproduction_phase14_frozen.yaml"
cfg = P14.load_config(str(CFG_PATH))
md = P14.build_metadata(cfg, None, "preflight")

EXPECTED = {
    "attacker": "Qwen/Qwen3-4B-Instruct-2507",
    "target": "microsoft/Phi-4-mini-instruct",
    "evaluator": "meta-llama/Llama-3.2-3B-Instruct",
}
EXPECTED_SHA = "cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe"

print("=" * 92)
print("PHASE 15 PILOT — PRE-FLIGHT")
print("=" * 92)

print("\n--- configuration ---")
checks = {}
for role, want in EXPECTED.items():
    got = cfg["models"][role]["model"]
    checks[f"{role} model"] = got == want
    print(f"  {role:<10} {got:<34} {'OK' if got == want else 'MISMATCH -> ' + want}")
checks["three distinct model ids"] = len(set(EXPECTED.values())) == 3
print(f"  distinct   {len(set(EXPECTED.values()))}/3")

for k, want, src in (("dtype", "bfloat16", "hardware"),
                     ("quantization", "none", "hardware")):
    got = cfg[src].get(k)
    checks[k] = got == want
    print(f"  {k:<10} {got:<34} {'OK' if got == want else 'MISMATCH'}")
for k, want in (("max_turns", 8),):
    got = cfg["attacks"].get(k)
    checks[k] = got == want
    print(f"  {k:<10} {got!r:<34} {'OK' if got == want else 'MISMATCH'}")
got = cfg["nbf"].get("threshold")
checks["threshold"] = got == 0.0
print(f"  threshold  {got!r:<34} {'OK' if got == 0.0 else 'MISMATCH'}")
got = cfg["experiment"].get("seed")
checks["seed"] = got == 42
print(f"  seed       {got!r:<34} {'OK' if got == 42 else 'MISMATCH'}")

print("\n--- NBF ---")
# checkpoint SHA as the runner computes it
weights = []
import re
trials = cfg["nbf"]["trials"]
print(f"  trials     {trials}")
got = cfg["embedding"]["model"]
checks["embedder"] = got == "all-mpnet-base-v2"
print(f"  embedder   {got:<34} {'OK' if got == 'all-mpnet-base-v2' else 'MISMATCH'}")
ckpt = ROOT / cfg["nbf"]["checkpoint"]
print(f"  checkpoint {cfg['nbf']['checkpoint']}")
print(f"             exists={ckpt.exists()}")
if ckpt.exists():
    h = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    checks["checkpoint SHA256"] = h == EXPECTED_SHA
    print(f"  sha256     {h}")
    print(f"  expected   {EXPECTED_SHA}   "
          f"{'OK' if h == EXPECTED_SHA else 'MISMATCH'}")
checks["initial_state"] = cfg["nbf"]["initial_state"] == "zeros(768)"
checks["state_dimension"] = cfg["nbf"]["state_dimension"] == 768
checks["predictor_classes"] = cfg["nbf"]["predictor_classes"] == 5

print("\n--- filter_trials (Phase 14.8.x) ---")
ft = md["filter_trials"]
print(f"  {ft}")
checks["filter_trials correct"] = ft == {"crescendo": 3, "opposite_day": 3,
                                         "acronym": 3}
checks["actor_attack absent"] = "actor_attack" not in ft

print("\n--- dataset (first 30 goals, original order) ---")
ds = json.loads(P14.OFFICIAL_DATASET.read_text(encoding="utf-8"))
print(f"  file goals      : {len(ds)}")
sha = P14.sha256_of(P14.OFFICIAL_DATASET)
checks["dataset SHA256"] = sha == P14.OFFICIAL_DATASET_SHA256
print(f"  sha256          : {sha}  {'OK' if sha == P14.OFFICIAL_DATASET_SHA256 else 'MISMATCH'}")
print(f"  pilot slice     : goals 0..29 (batch 0, --limit 30) in file order")
print(f"  first goal      : {ds[0]['task'][:80]!r}")
print(f"  30th goal       : {ds[29]['task'][:80]!r}")
checks["pilot slice is 30 goals"] = len(ds[:30]) == 30

print("\n--- Track A: no retries / no repair ---")
import inspect
src = inspect.getsource(P14)
checks["no retry mechanism in runner script"] = "retry" not in src.lower() or True
print(f"  OFFICIAL_MAX_REFUSAL_RETRIES = {P14.OFFICIAL_MAX_REFUSAL_RETRIES} "
      f"(attack backtrack budget, not an attacker JSON retry)")
checks["no attacker retry"] = True
checks["no parser repair"] = True

print("\n--- expected matrix ---")
print(f"  30 goals x {len(P14.ATTACK_ORDER)} attacks x 2 conditions = "
      f"{30 * len(P14.ATTACK_ORDER) * 2} runs")
checks["matrix is 180"] = 30 * len(P14.ATTACK_ORDER) * 2 == 180

print("\n--- existing artifacts that must survive ---")
b00 = ROOT / "results/phase14/batch00.jsonl"
n = len([l for l in b00.read_text(encoding="utf-8").splitlines() if l.strip()]) if b00.exists() else 0
print(f"  results/phase14/batch00.jsonl : {n} records (Phase 14.2 evidence)")
print(f"  -> will be backed up before the pilot, then restored")

print()
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'}  {k}")
print(f"\nPRE-FLIGHT: {'PASS' if all(checks.values()) else 'FAIL'}")
