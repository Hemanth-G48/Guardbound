#!/usr/bin/env python3
"""Explicitly-invoked paid connectivity check for the OpenAI cloud provider.

This is a standalone diagnostic, not a research entry point, and it never runs
implicitly: it requires ``--confirm-paid-request`` (``--dry-run`` performs the
same preconditions with zero spend). When the target is the frozen R1 model
(``gpt-4o-2024-08-06``), the centralized paid-authorization gate also applies;
the non-R1 default is not an R1 experiment.

What it does, in order
----------------------
1. Validate that the API key exists — without printing, storing or logging it.
2. Send one minimal text request to the configured model.
3. Verify the reply is non-empty and record its shape.
4. Send one structured-JSON request and verify the JSON contract.
5. Capture token usage, request duration, request id, and the model identifier.
6. Estimate cost from the rate card and label the estimate's basis.
7. Exit non-zero on any failure, cleanly otherwise.

Usage
-----
    # Zero-spend precondition check
    python scripts/cloud_smoke_test.py --dry-run

    # The one paid request pair (two small completions)
    python scripts/cloud_smoke_test.py --confirm-paid-request

    # Pin the exact snapshot
    python scripts/cloud_smoke_test.py --confirm-paid-request \\
        --model gpt-4o-mini-2024-07-18
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import _cloud_common as common  # noqa: E402

SMOKE_TEXT_PROMPT = "Reply with exactly one word: ready"
SMOKE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "value": {"type": "integer"},
    },
    "required": ["status", "value"],
    "additionalProperties": False,
}
SMOKE_JSON_PROMPT = (
    'Return a JSON object with exactly two fields: "status" set to the string '
    '"ok" and "value" set to the integer 1.'
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paid OpenAI smoke test (requires explicit confirmation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini-2024-07-18",
        help="Exact snapshot id (default: gpt-4o-mini-2024-07-18, the "
             "Configuration B snapshot). Avoid the floating alias.",
    )
    parser.add_argument(
        "--api-key-env", default="OPENAI_API_KEY",
        help="Environment variable holding the key (never the key itself).",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--rate-card", default=common.DEFAULT_RATE_CARD,
        help="Pricing file used for the cost estimate.",
    )
    parser.add_argument(
        "--structured-output", default="json_object",
        choices=["json_object", "json_schema", "none"],
        help="JSON mode for step 4 (json_object = official reference mode).",
    )
    parser.add_argument("--out", default=None, help="Write a JSON report here.")
    parser.add_argument(
        "--confirm-paid-request", action="store_true",
        help="Required to actually send requests. Without it, nothing is sent.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate key presence, pricing and config with zero API calls.",
    )
    common.add_paid_authorization_args(parser)
    return parser.parse_args(argv)


def _print_kv(label: str, value) -> None:
    print(f"  {label:<22} {value}")


def preflight(args: argparse.Namespace):
    """Everything that can be checked without spending money."""
    cfg = {
        "cloud": {"api_key_env": args.api_key_env, "rate_card": args.rate_card},
    }
    key_status = common.require_api_key(cfg)
    ledger = common.build_ledger(cfg, rate_card_path=args.rate_card)

    print("Preconditions")
    _print_kv("API key", f"{args.api_key_env}: {key_status}")
    _print_kv("model", args.model)
    _print_kv("rate card", ledger.rate_card.version or "(unversioned)")
    _print_kv("rate card verified", ledger.rate_card.verified)
    for warning in common.pricing_warnings(ledger, [args.model]):
        print(f"  WARNING: {warning}")
    return ledger


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        preflight(args)
        print("\nDRY RUN — no request sent, nothing billed.")
        return 0

    if not args.confirm_paid_request:
        print(
            "\nRefusing to send a paid request.\n"
            "Re-run with --confirm-paid-request to send it, or --dry-run to "
            "check preconditions only.",
            file=sys.stderr,
        )
        return 2

    # Phase 43 (P0-1): R1 targets require the centralized paid authorization
    # BEFORE the key is read or the client is constructed. Non-R1 targets
    # (the default gpt-4o-mini) are unaffected.
    common.guard_paid_model(args.model, args)

    ledger = preflight(args)

    from guardbound.llm.cloud_errors import CloudProviderError

    client = common.build_paid_chat_llm(
        args.model,
        args=args,
        model_kwargs={
            "api_key_env": args.api_key_env,
            "base_url": args.base_url,
            "timeout_s": args.timeout_s,
            "structured_output": args.structured_output,
            "json_schema": (
                SMOKE_JSON_SCHEMA
                if args.structured_output == "json_schema" else None),
            "json_schema_name": "smoke_test",
            "role": "smoke_test",
            "ledger": ledger,
        },
    )

    failures: list[str] = []

    print("\n[1/2] minimal text request")
    try:
        text = client.generate(
            [{"role": "user", "content": SMOKE_TEXT_PROMPT}], temperature=0.0
        )
        stats = client.last_request_stats
        _print_kv("reply", repr(common.safe(text))[:70])
        _print_kv("request id", stats.get("request_id"))
        _print_kv("finish reason", stats.get("finish_reason"))
        _print_kv("tokens in/out", f"{stats.get('input_tokens')}/{stats.get('output_tokens')}")
        _print_kv("duration (s)", stats.get("duration_s"))
        if not str(text or "").strip():
            failures.append("text request returned an empty response")
    except CloudProviderError as exc:
        print(f"  FAILED: {exc.to_dict()}", file=sys.stderr)
        failures.append(f"text request failed: {exc.category}")

    print("\n[2/2] structured JSON request")
    try:
        payload = client.generate(
            [{"role": "user", "content": SMOKE_JSON_PROMPT}],
            temperature=0.0,
            json_format=True,
        )
        stats = client.last_request_stats
        _print_kv("reply", repr(common.safe(payload))[:70])
        _print_kv("parsed as JSON", isinstance(payload, dict))
        _print_kv("request id", stats.get("request_id"))
        _print_kv("tokens in/out", f"{stats.get('input_tokens')}/{stats.get('output_tokens')}")
        _print_kv("duration (s)", stats.get("duration_s"))
        _print_kv("outcome", stats.get("outcome"))
        if not isinstance(payload, dict):
            failures.append(
                "structured request did not return a JSON object "
                f"(outcome={stats.get('outcome')})"
            )
    except CloudProviderError as exc:
        print(f"  FAILED: {exc.to_dict()}", file=sys.stderr)
        failures.append(f"structured request failed: {exc.category}")

    summary = ledger.summary()
    print("\nUsage and cost")
    _print_kv("model", args.model)
    _print_kv("calls", summary["calls"])
    _print_kv("tokens in", summary["input_tokens"])
    _print_kv("tokens out", summary["output_tokens"])
    _print_kv("total tokens", summary["total_tokens"])
    _print_kv("total duration (s)", summary["duration_s"])
    _print_kv("estimated cost (USD)", summary["estimated_cost_usd"])
    _print_kv("cost basis", summary["cost_basis"])
    _print_kv("unpriced calls", summary["unpriced_calls"])

    report = {
        "smoke_test": {
            "model": args.model,
            "api_key_env": args.api_key_env,
            "structured_output": args.structured_output,
            "rate_card": ledger.rate_card.to_dict(),
        },
        "environment": common.environment_info(),
        "git_commit": common.git_commit(),
        "usage": summary,
        "records": [r.to_dict() for r in ledger.records()],
        "failures": failures,
        "result": "PASS" if not failures else "FAIL",
    }
    if args.out:
        common.write_json(args.out, report)
        print(f"\nReport written to {args.out}")

    print(f"\nSMOKE TEST: {'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"  - {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
