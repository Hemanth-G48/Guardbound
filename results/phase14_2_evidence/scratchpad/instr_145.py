"""Instrumentation verification for Phase 14.5.

The qualification rows record `structured_output_mode: None` because the
production contract functions call generate() without passing it. That is the
caller's argument — NOT the mode in effect. The backend falls back to its
constructor default. This proves what was actually active.

No evaluator contract is invoked; this is instrumentation verification only,
which Step 15 permits.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402
from guardbound.llm.model_manager import ModelManager  # noqa: E402
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
out = {}

mgr = ModelManager(device="cuda")
llm = build_role_llm(
    {"provider": "local", "model": MODEL, "max_new_tokens": 256},
    "evaluator", manager=mgr, structured_output_mode="constrained_json",
)

print(f"declared on backend      : {llm.structured_output_mode!r}")
out["effective_structured_output_mode"] = llm.structured_output_mode
print(f"resolved on backend      : {llm.structured_output_mode!r}")

mgr.ensure_resident("evaluator")
pipe = llm._inner._pipeline
seen = {}
orig = pipe.model.generate


def spy(*a, **kw):
    seen["logits_processor"] = "logits_processor" in kw
    seen["stopping_criteria"] = "stopping_criteria" in kw
    seen["max_new_tokens"] = kw.get("max_new_tokens")
    seen["do_sample"] = kw.get("do_sample")
    seen["temperature"] = kw.get("temperature")
    return orig(*a, **kw)


pipe.model.generate = spy

# Call generate the same way the production contracts do: WITHOUT passing
# structured_output_mode, so the constructor default must supply it.
res = llm.generate(
    [{"role": "user", "content": 'Reply with JSON only: {"ok": true}'}],
    temperature=0.0, json_format=True,
)
print(f"generation kwargs seen   : {seen}")
print(f"return type              : {type(res).__name__}")
out["generation_kwargs_seen"] = seen
out["return_type"] = type(res).__name__
out["logits_processor_attached"] = seen.get("logits_processor", False)
out["stopping_criteria_attached"] = seen.get("stopping_criteria", False)
out["verdict"] = (
    "constrained_json was ACTIVE: the backend resolved it from its constructor "
    "default even though callers pass structured_output_mode=None"
    if seen.get("logits_processor") else
    "constrained_json was NOT attached - the decoder would not have constrained generation"
)
print(f"\n{out['verdict']}")

mgr.unload_all()
torch.cuda.empty_cache()
print(f"residency released: {mgr.residency_snapshot()}")
Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14_5_instrumentation_check.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("wrote results/phase14_5_instrumentation_check.json")
