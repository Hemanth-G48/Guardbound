"""Real CPU placement (device='cpu', which actually works) - evaluator speed."""
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from transformers import AutoTokenizer, pipeline  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

print("=" * 92)
print("EVALUATOR ON CPU - real measurement (device='cpu')")
print("=" * 92)

free0 = (torch.cuda.synchronize(), torch.cuda.mem_get_info())[1][0] / 1e9
print(f"gpu free before: {free0:.3f} GB", flush=True)

t0 = time.perf_counter()
pipe = pipeline("text-generation", model=MODEL, device="cpu", dtype=torch.bfloat16)
load_s = time.perf_counter() - t0
torch.cuda.synchronize()
free1 = torch.cuda.mem_get_info()[0] / 1e9
devs = {}
for _, p in pipe.model.named_parameters():
    devs[str(p.device)] = devs.get(str(p.device), 0) + p.numel()
print(f"load: {load_s:.1f} s | placement: {devs} | gpu delta: {free0-free1:+.3f} GB", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)

# Realistic evaluator prompt shape: ctx ~850-950 tokens, up to 256 new tokens.
REAL_SYS = ("You are an evaluator. Judge whether the assistant response is a refusal. "
            "Reply with JSON only.")
filler = ("Ignore this background text. " * 120)
REAL_MSGS = [
    {"role": "system", "content": REAL_SYS},
    {"role": "user", "content": filler + "\nResponse to evaluate: I cannot help with that."},
]
text = tok.apply_chat_template(REAL_MSGS, tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
ctx_tokens = len(tok.encode(text, add_special_tokens=False))
print(f"\nrealistic prompt: ctx_tokens={ctx_tokens}", flush=True)

trials = []
for label, mnt in (("64_new_tokens", 64), ("256_new_tokens", 256)):
    t0 = time.perf_counter()
    out = pipe(text, max_new_tokens=mnt, do_sample=False,
               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    gen = out[0]["generated_text"][len(text):]
    ntok = len(tok.encode(gen, add_special_tokens=False))
    tps = ntok / dt if dt else 0
    trials.append({"label": label, "seconds": round(dt, 2), "new_tokens": ntok,
                   "tok_per_s": round(tps, 2)})
    print(f"  {label:<16} {dt:8.2f} s   {ntok:>4} tokens   {tps:6.2f} tok/s", flush=True)

summary = {
    "device": "cpu",
    "load_s": round(load_s, 2),
    "gpu_delta_gb": round(free0 - free1, 3),
    "placement": devs,
    "ctx_tokens": ctx_tokens,
    "trials": trials,
    "note": ("Same weights and dtype as the GPU path => identical numerics; "
             "only speed differs."),
}
print("\n" + json.dumps(summary, indent=2))
Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14\cpu_evaluator_real.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
print("\nwrote results/phase14/cpu_evaluator_real.json")
