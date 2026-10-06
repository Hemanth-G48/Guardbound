"""Phase 16.5 F1 probe: does the target stop naturally (EOS), and at what length?

OBSERVATION ONLY. Uses a generous probe budget plus a wall-clock stopping
criterion so a rambling generation cannot hang the probe; neither is a production
setting. The question is whether natural EOS terminates generation well below any
probe budget.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path("C:/Users/CSLAB/Desktop/hemanth/Guardbound")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import MaxTimeCriteria, StoppingCriteriaList  # noqa: E402

from guardbound.llm.model_manager import ModelManager  # noqa: E402
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402

PROBE_TOKENS = int(os.environ.get("PROBE_TOKENS", "2048"))
PROBE_SECONDS = float(os.environ.get("PROBE_SECONDS", "45"))
N = int(os.environ.get("N", "8"))

cfg = yaml.safe_load(
    (ROOT / "configs/reproduction_phase16_5_forensics.yaml").read_text(encoding="utf-8"))
recs = [json.loads(l) for l in (ROOT / "results/phase15_pilot_30/trackA_frozen/"
                                "batch00_goals000-029_180records.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
recs.sort(key=lambda r: (r["goal_id"], r["attack"]))

cases = []
for r in recs:
    turns = r.get("turns") or []
    if len(turns) >= 4:
        cases.append((r, 1))
    if len(cases) >= N:
        break

manager = ModelManager(device="cuda")
llm = build_role_llm(cfg["models"]["target"], "target", manager=manager,
                     structured_output_mode="constrained_json")
inner = llm._inner
manager.set_run_context("P16_5_F1_PROBE")
t0 = time.time()
manager.ensure_resident("target")
print(f"loaded in {time.time()-t0:.1f}s; probe budget={PROBE_TOKENS} tokens / "
      f"{PROBE_SECONDS}s", flush=True)
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

tok = inner._get_tokenizer()
model = inner._get_pipeline().model
dev = next(model.parameters()).device
eos = tok.eos_token_id
eos = eos if isinstance(eos, list) else [eos]

rows = []
for i, (r, turn_idx) in enumerate(cases):
    turns = r["turns"]
    msgs = [{"role": "system", "content": r.get("target_system") or "You are a helpful assistant."}]
    for t in turns[:turn_idx]:
        msgs.append({"role": "user", "content": t["query"]})
        msgs.append({"role": "assistant", "content": t["response"]})
    msgs.append({"role": "user", "content": turns[turn_idx]["query"]})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   **inner.chat_template_kwargs)
    inputs = {k: v.to(dev) for k, v in
              tok(text, return_tensors="pt", truncation=True,
                  max_length=131072).items()}
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=PROBE_TOKENS, do_sample=True,
            use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            temperature=0.7, top_p=1.0, top_k=0,
            stopping_criteria=StoppingCriteriaList([MaxTimeCriteria(PROBE_SECONDS)]),
        )
    dt = time.time() - t0
    new = out[0][inputs["input_ids"].shape[1]:]
    last = int(new[-1]) if len(new) else -1
    ended_eos = last in eos
    reply = tok.decode(new, skip_special_tokens=True)
    rec_tokens = len(tok.encode(turns[turn_idx]["response"], add_special_tokens=False))
    rows.append({"goal_id": r["goal_id"], "attack": r["attack"], "turn": turn_idx,
                 "input_tokens": int(inputs["input_ids"].shape[1]),
                 "new_tokens": int(len(new)), "ended_on_eos": ended_eos,
                 "probe_tokens": PROBE_TOKENS, "seconds": round(dt, 1),
                 "recorded_tokens": rec_tokens,
                 "midword": not reply.rstrip().endswith((".", "!", "?", '"', ")"))})
    print(f"[probe {i}] goal={r['goal_id']} turn={turn_idx}: "
          f"recorded={rec_tokens} tok -> natural={len(new)} tok "
          f"eos={ended_eos} {dt:.0f}s midword={rows[-1]['midword']}", flush=True)

n = [x["new_tokens"] for x in rows]
rec = [x["recorded_tokens"] for x in rows]
print()
print(f"cases: {len(rows)}")
print(f"natural tokens : mean={statistics.mean(n):.0f} median={statistics.median(n)} "
      f"min={min(n)} max={max(n)}")
print(f"recorded tokens: mean={statistics.mean(rec):.0f} median={statistics.median(rec)} "
      f"max={max(rec)}")
print(f"ended on EOS   : {sum(1 for x in rows if x['ended_on_eos'])}/{len(rows)}")
print(f"hit probe cap  : {sum(1 for x in rows if x['new_tokens'] >= PROBE_TOKENS)}/{len(rows)}")
print(f"peak vram      : {(torch.cuda.max_memory_allocated()/1e9):.2f} GB")
out_path = ROOT / "results/phase16_5_forensics/f1_natural_stopping_probe.json"
out_path.write_text(json.dumps({"probe_tokens": PROBE_TOKENS,
                                "probe_seconds": PROBE_SECONDS,
                                "rows": rows}, indent=2), encoding="utf-8")
manager.unload_all()
print("wrote", out_path)
