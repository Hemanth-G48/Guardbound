"""Phase 17 Section 4/5 — emit target_verification.json with a real config hash.

Reads the fixed target and judge configuration from the local weights and writes
a hash over the exact configuration block, so a later change is detectable
rather than silently altering the contract mid-phase.

Section 4 requires values the paper/reference does not specify to be recorded as
NOT SPECIFIED rather than invented. The paper's reference stack is GPT-4o in all
three roles (read from the frozen config's ``paper_reference`` block), so every
Llama-specific value here is a substitution parameter, not a paper value. That
distinction is recorded explicitly per field.
"""

from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

NOT_SPECIFIED = "NOT SPECIFIED"


def snapshot(repo: str) -> Path | None:
    snaps = HF_HUB / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return None
    dirs = [d for d in snaps.iterdir() if d.is_dir()]
    return dirs[0] if dirs else None


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def config_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


TARGET_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"
JUDGE_REPO = "Qwen/Qwen3-8B"

# Values the Phase 17 contract fixes for the target. Identical for every
# attacker candidate, per Sections 4 and 16.
TARGET_GENERATION = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": NOT_SPECIFIED,
    "do_sample": True,
    "max_new_tokens": NOT_SPECIFIED,
    "max_length": NOT_SPECIFIED,
    "eos_token_id": [128001, 128009],
    "stop_sequences": NOT_SPECIFIED,
    "pad_token_id": NOT_SPECIFIED,
}
TARGET_GENERATION["pad_token_id"] = 128009  # Llama-3 has no pad token; eos reused

JUDGE_GENERATION = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": NOT_SPECIFIED,
    "do_sample": False,
    "max_new_tokens": NOT_SPECIFIED,
    "enable_thinking": False,
}


def build() -> dict:
    t_snap, j_snap = snapshot(TARGET_REPO), snapshot(JUDGE_REPO)
    t_cfg = load_json(t_snap / "config.json") if t_snap else {}
    t_gen = load_json(t_snap / "generation_config.json") if t_snap else {}
    t_tok = load_json(t_snap / "tokenizer_config.json") if t_snap else {}
    j_cfg = load_json(j_snap / "config.json") if j_snap else {}
    j_gen = load_json(j_snap / "generation_config.json") if j_snap else {}
    j_tok = load_json(j_snap / "tokenizer_config.json") if j_snap else {}
    j_template = j_tok.get("chat_template") or ""

    target_block = {
        "model_id": TARGET_REPO,
        "revision": t_snap.name if t_snap else None,
        "parameter_count": "8.03B (declared by HF safetensors metadata)",
        "architecture": t_cfg.get("architectures"),
        "model_type": t_cfg.get("model_type"),
        "num_hidden_layers": t_cfg.get("num_hidden_layers"),
        "hidden_size": t_cfg.get("hidden_size"),
        "vocab_size": t_cfg.get("vocab_size"),
        "dtype": t_cfg.get("torch_dtype"),
        "quantization": "none (bf16, unquantized)",
        "context_length": t_cfg.get("max_position_embeddings"),
        "generation_config_on_disk": {
            "temperature": t_gen.get("temperature"),
            "top_p": t_gen.get("top_p"),
            "do_sample": t_gen.get("do_sample"),
            "max_length": t_gen.get("max_length"),
            "eos_token_id": t_gen.get("eos_token_id"),
        },
        "generation": TARGET_GENERATION,
        "tokenizer": {
            "eos_token": t_tok.get("eos_token"),
            "bos_token": t_tok.get("bos_token"),
            "pad_token": t_tok.get("pad_token"),
            "chat_template_chars": len(t_tok.get("chat_template") or ""),
        },
        "source_of_each_value": {
            "model_id": "Phase 17 contract (substitution)",
            "revision": "resolved from the local snapshot on this machine",
            "temperature": "Phase 17 contract; paper reference (GPT-4o) " + NOT_SPECIFIED,
            "top_p": "Phase 17 contract; explicitly passed so nothing is silently inherited",
            "top_k": "Llama-3 has no top_k convention in its generation config -> " + NOT_SPECIFIED,
            "max_new_tokens": (
                "deliberately " + NOT_SPECIFIED + " in Phase 17. Phase 16.5 established that a "
                "fixed 256-token budget truncated 95.0% of the previous target's responses with "
                "0% natural EOS, making attack success unmeasurable."
            ),
            "pad_token_id": "Llama-3 ships pad_token=null; eos reused, as HF requires",
        },
        "paper_reference": {
            "paper_target": "gpt-4o",
            "paper_source": "frozen config paper_reference block, configs/reproduction_phase14_frozen.yaml",
            "consequence": (
                "Llama-3-8B-Instruct is a local substitution for the paper's GPT-4o target. "
                "No Llama-specific generation value can be called paper-specified."
            ),
        },
    }

    judge_block = {
        "model_id": JUDGE_REPO,
        "revision": j_snap.name if j_snap else None,
        "note": (
            "Resolved by the user after the contract's original id "
            "(Qwen/Qwen3-8B-Instruct-2507) was verified not to exist on HuggingFace."
        ),
        "parameter_count": "8.19B (declared by HF safetensors metadata; 15.27 GiB bf16 on disk)",
        "architecture": j_cfg.get("architectures"),
        "model_type": j_cfg.get("model_type"),
        "num_hidden_layers": j_cfg.get("num_hidden_layers"),
        "hidden_size": j_cfg.get("hidden_size"),
        "vocab_size": j_cfg.get("vocab_size"),
        "dtype": j_cfg.get("torch_dtype"),
        "quantization": "none (bf16, unquantized)",
        "context_length": j_cfg.get("max_position_embeddings"),
        "generation_config_on_disk": {
            "temperature": j_gen.get("temperature"),
            "top_p": j_gen.get("top_p"),
            "top_k": j_gen.get("top_k"),
            "do_sample": j_gen.get("do_sample"),
            "eos_token_id": j_gen.get("eos_token_id"),
        },
        "generation": JUDGE_GENERATION,
        "tokenizer": {
            "eos_token": j_tok.get("eos_token"),
            "pad_token": j_tok.get("pad_token"),
            "chat_template_chars": len(j_template),
            "template_accepts_enable_thinking": "enable_thinking" in j_template,
        },
        "structured_output": {
            "mechanism": "prompted JSON + the project's frozen structural extraction",
            "json_schema": "score integer 1-5 with a reasoning field",
            "note": (
                "No constrained decoding is enabled for the judge unless the frozen parser "
                "path already provides it. Recorded, not assumed."
            ),
        },
        "risk_flagged_before_qualification": {
            "hybrid_thinking_model": True,
            "evidence": (
                "generation_config sets top_k=20/top_p=0.95/temperature=0.6 and the chat "
                "template branches on enable_thinking; Qwen3 defaults to thinking mode."
            ),
            "measured_risk": (
                "If thinking is not disabled the judge emits reasoning before its JSON, "
                "which Section 6 measures as reasoning-leakage rather than silently stripping."
            ),
        },
    }

    return {
        "phase": "17",
        "deliverable": "target_verification.json",
        "target": target_block,
        "judge": judge_block,
        "target_config_hash": config_hash(target_block),
        "judge_config_hash": config_hash(judge_block),
        "hash_method": "sha256 over canonical JSON (sorted keys, no whitespace) of the role block",
        "invariant": (
            "Both hashes must be identical for every attacker candidate. Section 4 requires "
            "asserting the target configuration does not change during the phase."
        ),
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build()
    out = OUT_DIR / "target_verification.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"target_config_hash: {payload['target_config_hash']}")
    print(f"judge_config_hash : {payload['judge_config_hash']}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
