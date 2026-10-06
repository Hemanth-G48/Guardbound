"""Phase 16.5 §7 — F3 validation: prove the effective generation parameters.

No GPU, no model load: builds the role backends and asks each what it will pass to
generate(), then proves the kwargs by driving generate() through a fake pipeline.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path("C:/Users/CSLAB/Desktop/hemanth/Guardbound")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from guardbound.llm.model_manager import ModelManager  # noqa: E402
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402

FROZEN = ROOT / "configs/reproduction_phase14_frozen.yaml"
FORENSIC = ROOT / "configs/reproduction_phase16_5_forensics.yaml"


class FakePipe:
    def __init__(self, model):
        self.model = model


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    model_max_length = 131072

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT"

    def __call__(self, text, **kwargs):
        import torch
        return {"input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, ids, **kwargs):
        return "ok"


class FakeModel:
    def __init__(self):
        import torch
        self.config = type("C", (), {"max_position_embeddings": 131072})()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []

    def parameters(self):
        return iter([self._p])

    def eval(self):
        return self

    def generate(self, **kwargs):
        import torch
        self.calls.append(dict(kwargs))
        return torch.tensor([[1, 2, 3, 9, 10]])


def probe(cfg_path: Path, label: str) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"config": str(cfg_path.name), "roles": {}}
    for role in ("attacker", "target", "evaluator"):
        manager = ModelManager()
        llm = build_role_llm(cfg["models"][role], role, manager=manager,
                             structured_output_mode="constrained_json")
        inner = llm._inner
        pipe, model = FakePipe(FakeModel()), FakeModel()
        pipe.model = model
        inner._get_tokenizer = lambda: FakeTokenizer()
        inner._get_pipeline = lambda: pipe
        inner.generate([{"role": "user", "content": "hi"}], temperature=0.7,
                       json_format=False)
        kw = model.calls[-1]
        out["roles"][role] = {
            "reported": inner.generation_parameters(),
            "kwargs_passed_to_generate": {
                k: kw.get(k) for k in ("max_new_tokens", "top_p", "top_k",
                                       "do_sample", "temperature", "use_cache")},
            "top_p_present": "top_p" in kw,
            "top_k_present": "top_k" in kw,
        }
    return out


report = {"frozen_config": probe(FROZEN, "frozen"),
          "phase16_5_config": probe(FORENSIC, "forensics")}
out_path = ROOT / "results/phase16_5_forensics/f3_sampling_validation.json"
out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

for section, data in report.items():
    print("=" * 76)
    print(section)
    for role, r in data["roles"].items():
        print(f"  {role}:")
        print(f"    declared max_new_tokens : {r['reported']['max_new_tokens']} "
              f"(effective: {r['reported']['max_new_tokens_effective']})")
        print(f"    top_p   : {r['kwargs_passed_to_generate']['top_p']}  "
              f"({r['reported']['top_p_source']})")
        print(f"    top_k   : {r['kwargs_passed_to_generate']['top_k']}  "
              f"({r['reported']['top_k_source']})")
        print(f"    do_sample: {r['kwargs_passed_to_generate']['do_sample']}  "
              f"temp={r['kwargs_passed_to_generate']['temperature']}")
print("\nwrote", out_path)
