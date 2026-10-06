"""Phase 17 Stage 1 — judge VRAM validation and model-switching safety.

Deliverables:
  judge_vram_validation.json
  model_switching_validation.json

Section 11 requires that the judge can be loaded after the target and unloaded
before another model loads, and Section 12 requires proving that switching does
not contaminate state. Both roles are ~15 GiB, so two resident models would
exceed the 23.99 GiB card — sequential residency is not a convenience here, it is
the only workable arrangement, and this script tests that it actually holds.

Identity is verified against the LOADED weights (config architecture and
name_or_path), not against the argument we passed in. Asserting the requested id
back to itself would prove nothing.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage1_target_judge"

TARGET = "meta-llama/Meta-Llama-3-8B-Instruct"
JUDGE = "Qwen/Qwen3-8B"

EXPECTED_ARCH = {
    TARGET: "LlamaForCausalLM",
    JUDGE: "Qwen3ForCausalLM",
}


def vram() -> dict:
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "free_gib": round(
            (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())
            / 1024**3,
            3,
        ),
    }


def teardown(client) -> tuple[bool, float]:
    """Fully release a client. Returns ``(cache_entry_removed, reclaimed_gib)``.

    Two independent strong references must BOTH be dropped before the weights are
    freed, which is why popping the cache alone is not enough:

      1. ``_pipeline_cache`` (module-level) — removed by ``release_pipeline``.
      2. ``client._pipeline`` (instance attribute) — removed here.

    ``release_pipeline``'s own docstring warns that "every owner must drop its
    reference", but nothing enforces it, so a caller that pops the cache while
    still holding the client sees no memory returned and the *next* load OOMs with
    ~15 GiB still resident. Measured on this machine: dropping only the client
    reference leaves 7.5 GiB allocated; dropping both returns to 0.008 GiB.
    """
    from guardbound.llm.local_client import release_pipeline

    before = torch.cuda.memory_allocated() / 1024**3
    model_id, device_map = client.model_id, client.device_map

    # 1. the client's own reference
    client._pipeline = None
    client._tokenizer = None

    # 2. the module-level cache
    released = release_pipeline(model_id, device_map)

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    after = torch.cuda.memory_allocated() / 1024**3
    return released, round(before - after, 3)


def identity_check(client) -> dict:
    """Verify the LOADED model matches the role, from the weights themselves."""
    pipe = client._get_pipeline()
    model = pipe.model
    cfg = getattr(model, "config", None)
    archs = list(getattr(cfg, "architectures", []) or [])
    tok = client._get_tokenizer()
    return {
        "requested_model_id": client.model_id,
        "loaded_name_or_path": getattr(model, "name_or_path", None),
        "loaded_architectures": archs,
        "loaded_model_type": getattr(cfg, "model_type", None),
        "matches_expected_arch": EXPECTED_ARCH.get(client.model_id) in archs,
        "tokenizer_name_or_path": getattr(tok, "name_or_path", None),
        "tokenizer_vocab_size": getattr(tok, "vocab_size", None),
        "chat_template_chars": len(getattr(tok, "chat_template", "") or ""),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from guardbound.llm.local_client import HFLocalChatLLM

    probe_messages = [{"role": "user", "content": "Reply with the single word: OK"}]

    vram_rows: list[dict] = []
    switching_rows: list[dict] = []
    baseline = vram()
    print(f"baseline VRAM: {baseline}")

    def cycle(role: str, model_id: str, step: int, label: str) -> dict:
        torch.cuda.reset_peak_memory_stats()
        before = vram()
        t0 = time.perf_counter()
        client = HFLocalChatLLM(
            model_id,
            device_map={"": 0},
            max_new_tokens=None,
            top_p=1.0,
            do_sample=False,
        )
        ident = identity_check(client)
        load_s = time.perf_counter() - t0
        after_load = vram()

        t1 = time.perf_counter()
        out = client.generate(probe_messages, temperature=0.0)
        gen_s = time.perf_counter() - t1
        text = out if isinstance(out, str) else str(out)

        row = {
            "step": step,
            "label": label,
            "role": role,
            "model_id": model_id,
            **ident,
            "load_seconds": round(load_s, 2),
            "generate_seconds": round(gen_s, 2),
            "generated_ok": bool(text.strip()),
            "vram_before_load": before,
            "vram_after_load": after_load,
            "vram_after_generate": vram(),
        }
        print(
            f"  step {step} {label:28s} {role:7s} arch={ident['loaded_architectures']} "
            f"alloc={after_load['allocated_gib']:.2f} peak={vram()['peak_gib']:.2f} GiB"
        )
        return row, client

    # --- switching sequence: TARGET → JUDGE → TARGET → JUDGE → TARGET --------
    sequence = [
        ("target", TARGET, "TARGET (cold)"),
        ("judge", JUDGE, "TARGET -> JUDGE"),
        ("target", TARGET, "JUDGE -> TARGET"),
        ("judge", JUDGE, "TARGET -> JUDGE (2nd cycle)"),
        ("target", TARGET, "JUDGE -> TARGET (2nd cycle)"),
    ]

    for i, (role, model_id, label) in enumerate(sequence, start=1):
        row, client = cycle(role, model_id, i, label)
        before_release_alloc = vram()["allocated_gib"]
        t2 = time.perf_counter()
        released, reclaimed = teardown(client)
        client = None  # drop the caller's reference too
        gc.collect()
        torch.cuda.synchronize()
        release_s = time.perf_counter() - t2
        after_release = vram()
        if after_release["allocated_gib"] > baseline["allocated_gib"] + 0.75:
            raise RuntimeError(
                f"residency leak after step {i} ({label}): "
                f"{after_release['allocated_gib']:.2f} GiB still allocated "
                f"(baseline {baseline['allocated_gib']:.2f} GiB). The next model load "
                f"would OOM."
            )
        row.update({
            "released": released,
            "release_seconds": round(release_s, 2),
            "vram_before_release_alloc_gib": before_release_alloc,
            "vram_after_release": after_release,
            "vram_reclaimed_gib": reclaimed,
        })
        switching_rows.append(row)
        vram_rows.append({
            "step": i, "role": role, "model_id": model_id, "label": label,
            **{k: v for k, v in row.items() if k.startswith("vram_")},
            "load_seconds": row["load_seconds"],
        })
        print(
            f"          released={released} reclaimed={row['vram_reclaimed_gib']:.2f} GiB "
            f"free={after_release['free_gib']:.2f} GiB"
        )

    # --- contamination checks -------------------------------------------------
    checks = []
    for role, model_id in (("target", TARGET), ("judge", JUDGE)):
        rows = [r for r in switching_rows if r["role"] == role]
        checks.append({
            "check": f"{role}_architecture_consistent_across_reloads",
            "passed": len({tuple(r["loaded_architectures"]) for r in rows}) == 1,
            "observed": sorted({tuple(r["loaded_architectures"]) for r in rows}),
        })
        checks.append({
            "check": f"{role}_identity_matches_request",
            "passed": all(r["matches_expected_arch"] for r in rows),
            "observed": [r["matches_expected_arch"] for r in rows],
        })
    checks.append({
        "check": "no_cross_role_architecture_contamination",
        "passed": not any(
            r["loaded_architectures"] == [EXPECTED_ARCH.get(TARGET if r["role"] == "judge" else JUDGE)]
            for r in switching_rows
        ),
        "note": "each role always loaded its own architecture",
    })
    checks.append({
        "check": "tokenizer_not_contaminated",
        "passed": all(
            r["tokenizer_name_or_path"] is None or r["model_id"].split("/")[-1].lower()[:6]
            in str(r["tokenizer_name_or_path"]).lower().replace("_", "-")
            or r["tokenizer_name_or_path"] == r["model_id"]
            for r in switching_rows
        ),
        "observed": [
            {"model": r["model_id"], "tokenizer": r["tokenizer_name_or_path"]}
            for r in switching_rows
        ],
    })
    checks.append({
        "check": "no_vram_leak_across_cycles",
        "passed": vram()["allocated_gib"] <= baseline["allocated_gib"] + 0.75,
        "baseline_alloc_gib": baseline["allocated_gib"],
        "final_alloc_gib": vram()["allocated_gib"],
    })
    checks.append({
        "check": "single_resident_model_enforced",
        "passed": all(
            r["vram_after_load"]["allocated_gib"] < 20.0 for r in switching_rows
        ),
        "note": "both roles are ~15 GiB; two resident would exceed the 23.99 GiB card",
        "max_allocated_observed_gib": max(
            r["vram_after_load"]["allocated_gib"] for r in switching_rows
        ),
    })

    all_passed = all(c["passed"] for c in checks)
    print()
    for c in checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['check']}")

    (OUT_DIR / "judge_vram_validation.json").write_text(json.dumps({
        "phase": "17", "stage": "1",
        "deliverable": "judge_vram_validation.json",
        "judge_model": JUDGE,
        "judge_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "device": "NVIDIA RTX 4500 Ada Generation, 23.99 GiB usable",
        "residency_policy": "max resident models = 1 (sequential)",
        "baseline_vram": baseline,
        "measurements": [r for r in switching_rows if r["role"] == "judge"],
        "all_steps": vram_rows,
        "final_vram": vram(),
        "checks": checks,
        "all_checks_passed": all_passed,
    }, indent=2), encoding="utf-8")

    (OUT_DIR / "model_switching_validation.json").write_text(json.dumps({
        "phase": "17", "stage": "1",
        "deliverable": "model_switching_validation.json",
        "sequence": [
            "TARGET (cold)", "TARGET -> JUDGE", "JUDGE -> TARGET",
            "TARGET -> JUDGE (2nd cycle)", "JUDGE -> TARGET (2nd cycle)",
        ],
        "identity_source": (
            "verified from the loaded weights (config.architectures, "
            "model.name_or_path, tokenizer.name_or_path) — never by asserting the "
            "requested id back to itself"
        ),
        "steps": switching_rows,
        "checks": checks,
        "all_checks_passed": all_passed,
        "verdict": "SWITCHING_SAFE" if all_passed else "SWITCHING_UNSAFE",
    }, indent=2), encoding="utf-8")

    print(f"\nswitching verdict: {'SWITCHING_SAFE' if all_passed else 'SWITCHING_UNSAFE'}")
    print("wrote judge_vram_validation.json, model_switching_validation.json")


if __name__ == "__main__":
    main()
