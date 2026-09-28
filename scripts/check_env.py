import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ckpt = Path("nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/models/models_best_nbf_released.pth")
print(f"NBF checkpoint exists: {ckpt.exists()}")

ds = Path("nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json")
print(f"Dataset exists: {ds.exists()}")
if ds.exists():
    with open(ds) as f:
        records = json.load(f)
    print(f"Dataset records: {len(records)}")
    for i, r in enumerate(records[:3]):
        task = r.get("task", "")
        print(f"  g{i}: {task[:80]}...")

for model in ["Qwen/Qwen3-4B-Instruct-2507", "microsoft/Phi-4-mini-instruct"]:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model.replace("/", "--")) / "snapshots"
    if cache.exists():
        snapshots = [d for d in cache.iterdir() if d.is_dir()]
        print(f"{model}: cached snapshots: {len(snapshots)}")
        if snapshots:
            weights = list(snapshots[0].rglob("*.safetensors"))
            print(f"  weight files: {len(weights)}")
    else:
        print(f"{model}: NOT cached")

# Check for torch.inference_mode and use_cache
print("---")
print("Checking HFLocalChatLLM for inference_mode/eval/use_cache...")
import inspect
from guardbound.llm.local_client import HFLocalChatLLM
gen_src = inspect.getsource(HFLocalChatLLM.generate)
print("inference_mode in generate:", "inference_mode" in gen_src)
print("eval in generate:", ".eval()" in gen_src)
print("use_cache in generate:", "use_cache" in gen_src)
print("no_grad in generate:", "no_grad" in gen_src)

pipe_src = inspect.getsource(HFLocalChatLLM._get_pipeline)
print("use_cache in _build_gen_config:", "use_cache" in inspect.getsource(HFLocalChatLLM._build_gen_config))
