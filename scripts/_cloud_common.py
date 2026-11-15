#!/usr/bin/env python3
"""Shared plumbing for the cloud experiment scripts.

Used by ``cloud_smoke_test.py``, ``cloud_pilot.py`` and
``cloud_model_qualification.py``. Everything here is deliberately explicit:
each precondition (key present, checkpoint identity, config invariants)
fails loudly *before* the first billable request rather than being discovered
halfway through an experiment.

Nothing in this module makes a network call.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from guardbound.llm.cloud_errors import describe_api_key, redact  # noqa: E402
from guardbound.llm.openai_client import (  # noqa: E402
    STRUCTURED_OUTPUT_JSON_OBJECT,
    retry_policy_record,
)
from guardbound.llm.paid_authorization import (  # noqa: E402
    QualificationCapability,
    R1_MODEL_IDS,
    _authorized,
    _issue_qualification_capability,
    is_r1_model,
    require_construction_authorization,
)
from guardbound.llm.usage import (  # noqa: E402
    UsageLedger,
    load_rate_card,
)

DEFAULT_CONFIG = "configs/cloud_research_gpt4o_mini.yaml"
DEFAULT_RATE_CARD = "configs/cloud_pricing.yaml"
DEFAULT_DATASET = (
    "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/"
    "harmbench_tasks.json"
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cloud config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    for required in ("models", "attacks", "temperature"):
        if required not in cfg:
            raise ValueError(f"{path}: missing required section {required!r}")
    for role in ("attacker", "target", "evaluator"):
        if role not in cfg["models"]:
            raise ValueError(f"{path}: missing models.{role}")
    # Record the source identity so the qualification gate can bind a report to
    # the experiment configuration it claims to describe.
    cfg["_source_path"] = str(path)
    cfg["_source_sha256"] = sha256_of(path)
    return cfg


def resolve_variant(cfg: dict[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    """Return ``(variant_name, merged_variant_config)``.

    The variant block only overrides the keys it names, so both variants share
    one model/temperature/dataset/attack definition.
    """
    variants = cfg.get("variants") or {}
    if name not in variants:
        raise ValueError(
            f"unknown variant {name!r}; available: {sorted(variants)}"
        )
    return name, dict(variants[name] or {})


def assert_invariants(cfg: dict[str, Any]) -> list[str]:
    """Confirm the two variants differ only in NBF enablement.

    Returns the list of invariant paths that were checked. Raises when a variant
    overrides something outside ``nbf`` — that would silently break the
    controlled comparison the experiment claims to be.
    """
    variants = cfg.get("variants") or {}
    if len(variants) < 2:
        return []
    checked: list[str] = []
    overridden_outside_nbf: dict[str, list[str]] = {}
    for name, block in variants.items():
        extra = sorted(k for k in (block or {}) if k != "nbf" and k != "description")
        if extra:
            overridden_outside_nbf[name] = extra
    if overridden_outside_nbf:
        raise ValueError(
            "Variants must differ only in NBF enablement; found overrides: "
            f"{overridden_outside_nbf}"
        )
    checked.extend(cfg.get("invariants") or [])
    return checked


def nbf_settings(cfg: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cfg.get("nbf") or {})
    merged.update((variant or {}).get("nbf") or {})
    merged.setdefault("enabled", False)
    return merged


# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #

def require_api_key(cfg: dict[str, Any]) -> str:
    """Confirm the key exists. Never prints or returns it.

    Returns the ``present (len=N)`` descriptor so the caller can record that a
    credential was available without recording the credential.
    """
    env_name = str((cfg.get("cloud") or {}).get("api_key_env", "OPENAI_API_KEY"))
    key = os.environ.get(env_name)
    if not key:
        raise SystemExit(
            f"[cloud] {env_name} is not set (status: {describe_api_key(None)}).\n"
            f"        Export it in your shell (it is never read from, or written\n"
            f"        to, this repository), then re-run. Do not paste the key\n"
            f"        into a config file or a command line argument."
        )
    return describe_api_key(key)


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_nbf_checkpoint(nbf_cfg: dict[str, Any]) -> dict[str, Any]:
    """Pin the NBF identity before a run: path, sha256, dims, threshold.

    Refuses to continue when the digest differs from the expected one, because a
    different checkpoint is a different experiment.
    """
    checkpoint = nbf_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("nbf.checkpoint is required for the NBF-ON variant")
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"NBF checkpoint not found: {path}. The validated checkpoint lives "
            f"at nbf_original_stuff/.../models/models_best_nbf_released.pth"
        )
    actual = sha256_of(path)
    expected = nbf_cfg.get("checkpoint_sha256_expected")
    if expected and actual != expected:
        raise ValueError(
            f"NBF checkpoint sha256 mismatch: expected {expected}, got {actual}. "
            f"Refusing to run — the checkpoint identity is part of the "
            f"experiment definition."
        )
    return {
        "path": str(path),
        "sha256": actual,
        "sha256_expected": expected,
        "embedding_model": nbf_cfg.get("embedding_model"),
        "embedding_dim": nbf_cfg.get("embedding_dim"),
        "state_dim": nbf_cfg.get("state_dim"),
        "threshold": nbf_cfg.get("threshold"),
        "eta": nbf_cfg.get("eta"),
        "mode": nbf_cfg.get("mode"),
        "steer_target": nbf_cfg.get("steer_target"),
        "initial_state": nbf_cfg.get("initial_state"),
    }


# --------------------------------------------------------------------------- #
# Qualification gate (Phase 36, P1-1)
# --------------------------------------------------------------------------- #

QUALIFIED_VERDICT = "QUALIFIED"
QUALIFICATION_ROLES = ("attacker", "target", "evaluator")

# Phase 41 / Phase 40B. The qualification implementation identity is separate
# from the experiment configuration identity; both gate a paid R1 run.
QUALIFICATION_SCHEMA_VERSION = "qualification.v2"
QUALIFICATION_PROBE_SPEC_VERSION = "r1-probe.v2"
MANDATORY_PROBE_NAMES = (
    "attacker",
    "actor_attack_pre_attack",
    "target",
    "evaluator",
    "nbf_integration",
    "candidate_filter",
)
REVIEW_DECISIONS = (
    "accept_degradation",
    "adjust_extraction_prompt",
    "exclude_actor_attack",
)

# Phase 43/45. R1 paid authorization is centralized here so a paid entry point
# cannot construct an R1 client without a current report. The model identity is
# the exact frozen snapshot; aliases are not treated as R1.
R1_PAID_MODEL_IDS = R1_MODEL_IDS
R1_EXPERIMENT_CONFIG = _PROJECT_ROOT / "configs" / "paper_reproduction_r1_gpt4o.yaml"
DELIVERY_TYPE_REVIEW_THRESHOLD = 0.50
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


class QualificationError(RuntimeError):
    """The paid path cannot proceed without a valid QUALIFIED report.

    Phase 27 made the local qualification gate fail closed inside
    ``cloud_model_qualification.py``. Phase 36 extends the same verdict to the
    paid experiment entry point: no paid client may be constructed until a
    QUALIFIED report that matches this configuration has been verified.
    """


class PaidAuthorizationError(QualificationError):
    """An R1 paid entry point was reached without current authorization."""


def _effective_role_models(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the effective three-role model configuration.

    A present ``models`` mapping is authoritative. A single-model config may
    expose the same roles at top level; those are normalized into the same
    shape. Missing or empty role model IDs fail closed — the frozen R1 config is
    never substituted for a supplied configuration.
    """
    if not isinstance(cfg, dict):
        raise PaidAuthorizationError(
            "cannot establish the effective runtime role models: config is not "
            "a mapping")
    configured = cfg.get("models")
    if isinstance(configured, dict):
        source = configured
    else:
        source = cfg
    models: dict[str, dict[str, Any]] = {}
    single_model = not isinstance(configured, dict)
    for role in QUALIFICATION_ROLES:
        role_cfg = source.get(role)
        models[role] = dict(role_cfg) if isinstance(role_cfg, dict) else {}
    missing: list[str] = []
    for role in QUALIFICATION_ROLES:
        # The actual single-model schema has no evaluator.model. Its evaluator
        # role is the rubric model; the three-model schema keeps models.evaluator.model.
        if single_model and role == "evaluator":
            model = models[role].get("rubric_model")
        else:
            model = models[role].get("model")
        if not isinstance(model, str) or not model.strip():
            missing.append(role)
        else:
            models[role]["model"] = model.strip()
    if missing:
        raise PaidAuthorizationError(
            "cannot establish the effective runtime role models: missing "
            f"model IDs for {missing}. A paper-reproduction configuration must "
            "expose attacker, target, and evaluator model IDs; the frozen R1 "
            "configuration is not substituted for a supplied configuration."
        )
    return models


def expected_qualification_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    """The configuration fields a qualification report must match to be current."""
    models = _effective_role_models(cfg)
    temperatures = cfg.get("temperature") or {}
    return {
        "models": {
            role: models[role]["model"] for role in QUALIFICATION_ROLES
        },
        "prompt_variant": prompt_variant_of(cfg),
        "attacker_prompt_condition": attacker_arm_name(cfg),
        "temperatures": {
            role: float(temperatures.get(role, 0.7))
            for role in QUALIFICATION_ROLES
        },
        # Phase 38 (P2-4): the structured-output mode is part of the API contract
        # the probes exercise, so a report qualified under a different mode is
        # not evidence for this run. Phase 51: an omitted setting uses the client
        # default, which is part of the effective runtime identity.
        "structured_output": (
            (cfg.get("cloud") or {}).get("structured_output")
            or STRUCTURED_OUTPUT_JSON_OBJECT
        ),
        # Phase 41: the experiment configuration the report describes. The
        # pilot config loaded from disk carries its own hash.
        "experiment_config_sha256": cfg.get("_source_sha256"),
    }


def _qualification_report_parts(cfg: dict[str, Any]) -> tuple[Path, str, str]:
    models = cfg.get("models") or {}
    tag = str((models.get("attacker") or {}).get("model") or "model").replace("/", "_")
    arm = attacker_arm_name(cfg)
    directory = Path(
        (cfg.get("qualification") or {}).get("output_dir")
        or "results/qualification"
    )
    return directory, tag, arm


def qualification_report_glob(cfg: dict[str, Any]) -> str:
    directory, tag, arm = _qualification_report_parts(cfg)
    return str(directory / f"{tag}.{arm}.*.qualification.v2.json")


def _run_id_from_report(path: Path) -> str:
    """``<tag>.<arm>.<run_id>.qualification.v2.json`` -> ``run_id``."""
    parts = path.name.split(".")
    return parts[-4] if len(parts) >= 5 else path.stem


def _select_newest_report(matches: list[Path]) -> Path:
    """Newest by run id. Same-timestamp ambiguity is an error, never a guess."""
    if not matches:
        raise QualificationError("no qualification reports to select from")
    ordered = sorted(matches, key=lambda p: (_run_id_from_report(p), p.name))
    newest = ordered[-1]
    newest_run = _run_id_from_report(newest)
    peers = [p for p in ordered if _run_id_from_report(p) == newest_run]
    if len(peers) > 1:
        raise QualificationError(
            f"multiple v2 qualification reports share run_id {newest_run!r}: "
            + ", ".join(str(p) for p in peers)
            + ". Pass --qualification-report explicitly."
        )
    return newest


def resolve_qualification_report(
    cfg: dict[str, Any], out_dir: str | Path | None = None
) -> Path:
    """Resolve the newest canonical v2 report; never fall back to v1 or older."""
    directory, tag, arm = _qualification_report_parts(cfg)
    directory = Path(out_dir) if out_dir is not None else directory
    pattern = f"{tag}.{arm}.*.qualification.v2.json"
    matches = list(directory.glob(pattern))
    if not matches:
        raise QualificationError(
            f"no v2 qualification report matches {directory / pattern}. Run "
            f"`python scripts/cloud_model_qualification.py --config <config>` "
            f"first; legacy v1 reports are readable for audit but not eligible "
            f"for the R1 gate."
        )
    return _select_newest_report(matches)


def default_qualification_report_path(cfg: dict[str, Any]) -> Path:
    """Historical name for :func:`resolve_qualification_report`."""
    return resolve_qualification_report(cfg)


def require_qualification(
    cfg: dict[str, Any], report_path: str | Path | None = None
) -> dict[str, Any]:
    """Verify a v2 QUALIFIED report before any paid client is constructed.

    Fail-closed on: missing/unreadable report, legacy v1 or unknown schema,
    wrong probe-spec version, a verdict other than ``QUALIFIED``, an incomplete
    probe set, any mandatory probe that is not ``PASS``, and stale identities.
    Diagnostic probes are never consulted. If several reports exist, only the
    newest is considered; a validation failure never falls back to an older one.

    A supplied paper-reproduction config must also carry a loader-recorded
    source identity that still re-hashes to the source file.
    """
    if isinstance(cfg, dict) and cfg.get("paper_reproduction"):
        require_verified_source(cfg)
    if report_path:
        path = Path(report_path)
        if not path.exists():
            raise QualificationError(
                f"no qualification report at {path}. Run "
                f"`python scripts/cloud_model_qualification.py --config <config>` "
                f"first; a paid run is refused without a QUALIFIED verdict."
            )
    else:
        path = resolve_qualification_report(cfg)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(
            f"qualification report is unreadable or not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise QualificationError(
            f"qualification report must be a JSON object: {path} "
            f"(got {type(payload).__name__})"
        )

    # Phase 41 (C5): v1 is readable, never eligible. Check the implementation
    # identity before the verdict so a stale QUALIFIED-looking v1 report cannot
    # be mistaken for evidence.
    schema = payload.get("schema_version")
    if schema is None:
        raise QualificationError(
            f"legacy v1 qualification report (no schema_version): {path}. "
            f"It remains readable for audit but is not eligible for the R1 gate; "
            f"re-run the qualification harness to produce "
            f"{QUALIFICATION_SCHEMA_VERSION}."
        )
    if schema != QUALIFICATION_SCHEMA_VERSION:
        raise QualificationError(
            f"unsupported qualification schema_version {schema!r}: {path}. "
            f"Expected {QUALIFICATION_SCHEMA_VERSION!r}."
        )
    probe_spec = payload.get("probe_spec_version")
    if probe_spec is None:
        raise QualificationError(
            f"qualification report has no probe_spec_version: {path}. "
            f"Expected {QUALIFICATION_PROBE_SPEC_VERSION!r}."
        )
    if probe_spec != QUALIFICATION_PROBE_SPEC_VERSION:
        raise QualificationError(
            f"qualification report was produced by probe spec {probe_spec!r}, "
            f"not {QUALIFICATION_PROBE_SPEC_VERSION!r}: {path}."
        )

    # Phase 43 (P2-3): provenance is part of eligibility, not decoration.
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
        raise QualificationError(
            f"qualification report has a missing or invalid run_id "
            f"{run_id!r}; expected UTC %Y%m%dT%H%M%SZ: {path}"
        )
    try:
        datetime.strptime(run_id, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise QualificationError(
            f"qualification report run_id is not a real UTC timestamp: "
            f"{run_id!r} ({path})"
        ) from exc
    if not path.name.endswith(".qualification.v2.json"):
        raise QualificationError(
            f"qualification report filename is not canonical "
            f"(<tag>.<arm>.<run_id>.qualification.v2.json): {path.name}"
        )
    if _run_id_from_report(path) != run_id:
        raise QualificationError(
            f"qualification report run_id {run_id!r} does not match the "
            f"canonical filename {path.name!r}"
        )

    declared_verdict = payload.get("overall")
    triggers = payload.get("review_triggers")
    if not isinstance(triggers, dict):
        raise QualificationError(
            f"qualification report has no review_triggers mapping: {path}")
    delivery = triggers.get("delivery_type")
    if not isinstance(delivery, dict):
        raise QualificationError(
            f"qualification report has no review_triggers.delivery_type "
            f"mapping: {path}")
    threshold = delivery.get("threshold")
    observed = delivery.get("observed")
    triggered = delivery.get("triggered")
    if (isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 < float(threshold) <= 1.0
            or float(threshold) != DELIVERY_TYPE_REVIEW_THRESHOLD):
        raise QualificationError(
            f"review_triggers.delivery_type.threshold must equal the frozen "
            f"{DELIVERY_TYPE_REVIEW_THRESHOLD:.2f}: {threshold!r} ({path})")
    if declared_verdict == QUALIFIED_VERDICT:
        # Phase 45 (P1-1): a QUALIFIED report must carry a completed
        # observation. Null is only meaningful for a non-qualified/early-stop
        # report, which cannot pass this gate in any case.
        if (observed is None or isinstance(observed, bool)
                or not isinstance(observed, (int, float))):
            raise QualificationError(
                f"a QUALIFIED report requires a numeric "
                f"review_triggers.delivery_type.observed: {observed!r} ({path})")
        if not isinstance(triggered, bool):
            raise QualificationError(
                f"a QUALIFIED report requires a boolean "
                f"review_triggers.delivery_type.triggered: {triggered!r} ({path})")
    if observed is not None and (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not 0.0 <= float(observed) <= 1.0
    ):
        raise QualificationError(
            f"review_triggers.delivery_type.observed must be null or a number "
            f"in [0, 1]: {observed!r} ({path})")
    if triggered is not None and not isinstance(triggered, bool):
        raise QualificationError(
            f"review_triggers.delivery_type.triggered must be a boolean or "
            f"null: {triggered!r} ({path})")
    expected_triggered = (
        None if observed is None
        else bool(float(observed) < float(threshold))
    )
    if triggered != expected_triggered:
        raise QualificationError(
            f"review_triggers.delivery_type.triggered={triggered!r} is "
            f"inconsistent with observed={observed!r} and "
            f"threshold={threshold!r}: {path}"
        )

    verdict = declared_verdict
    if verdict != QUALIFIED_VERDICT:
        raise QualificationError(
            f"qualification verdict is {verdict!r}, not {QUALIFIED_VERDICT!r}: {path}"
        )
    if payload.get("probes_complete") is not True:
        raise QualificationError(
            f"qualification report did not complete its probe set "
            f"(probes_complete={payload.get('probes_complete')!r}): {path}"
        )

    statuses = payload.get("probe_status")
    if not isinstance(statuses, dict):
        raise QualificationError(
            f"qualification report has no probe_status mapping: {path}")
    for name in MANDATORY_PROBE_NAMES:
        entry = statuses.get(name)
        if not isinstance(entry, dict):
            raise QualificationError(
                f"mandatory probe {name!r} is missing from the report: {path}")
        if entry.get("status") != "PASS":
            raise QualificationError(
                f"mandatory probe {name!r} has status "
                f"{entry.get('status')!r}, not 'PASS': {path}"
            )

    expected = expected_qualification_identity(cfg)
    recorded = payload.get("configuration_under_qualification") or {}
    problems: list[str] = []
    if recorded.get("models") != expected["models"]:
        problems.append(
            f"models {recorded.get('models')!r} != {expected['models']!r}")
    if recorded.get("prompt_variant") != expected["prompt_variant"]:
        problems.append(
            f"prompt_variant {recorded.get('prompt_variant')!r} != "
            f"{expected['prompt_variant']!r}")
    if payload.get("attacker_prompt_condition") != expected["attacker_prompt_condition"]:
        problems.append(
            f"attacker_prompt_condition "
            f"{payload.get('attacker_prompt_condition')!r} != "
            f"{expected['attacker_prompt_condition']!r}")
    if recorded.get("temperatures") != expected["temperatures"]:
        problems.append(
            f"temperatures {recorded.get('temperatures')!r} != "
            f"{expected['temperatures']!r}")
    if recorded.get("structured_output") != expected["structured_output"]:
        problems.append(
            f"structured_output {recorded.get('structured_output')!r} != "
            f"{expected['structured_output']!r}")

    experiment_config = payload.get("experiment_config") or {}
    recorded_hash = experiment_config.get("sha256")
    if not recorded_hash:
        problems.append("experiment_config.sha256 is missing")
    else:
        recorded_path = experiment_config.get("path")
        if recorded_path:
            candidate = Path(recorded_path)
            if not candidate.is_absolute():
                candidate = _PROJECT_ROOT / candidate
            if candidate.exists() and sha256_of(candidate) != recorded_hash:
                problems.append(
                    f"experiment_config.sha256 does not match {recorded_path}")
        expected_hash = expected.get("experiment_config_sha256")
        if expected_hash and recorded_hash != expected_hash:
            problems.append(
                f"experiment_config.sha256 {recorded_hash[:12]}… != "
                f"{expected_hash[:12]}… for this configuration")

    if problems:
        raise QualificationError(
            "qualification report is stale for this configuration: "
            + "; ".join(problems)
            + f" ({path}). Re-run the qualification harness for these settings."
        )

    return {
        "report": str(path),
        "sha256": sha256_of(path),
        "schema_version": schema,
        "probe_spec_version": probe_spec,
        "run_id": run_id,
        "overall": verdict,
        "probes_complete": True,
        "attacker_prompt_condition": payload.get("attacker_prompt_condition"),
        "configuration_under_qualification": recorded,
        "experiment_config": experiment_config,
        "review_triggers": triggers,
        "gate": "enforced before paid client construction",
    }


def load_review_decision(path: str | Path) -> dict[str, Any]:
    """Read and validate the separate operator decision record.

    The qualification report is immutable and carries only the trigger; the
    decision lives in the experiment authorization/manifest.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(
            f"review decision is unreadable or not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise QualificationError(
            f"review decision must be a JSON object: {path}")
    required = (
        "qualification_report_sha256",
        "run_id",
        "configuration_sha256",
        "arm",
        "trigger",
        "decision",
        "rationale",
        "decided_by",
        "decided_at",
        "actor_attack_eligible",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise QualificationError(
            f"review decision is missing {missing}: {path}")
    for key in ("qualification_report_sha256", "configuration_sha256"):
        value = payload[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value.lower())):
            raise QualificationError(
                f"review decision {key} must be a 64-character hexadecimal "
                f"SHA-256: {path}")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
        raise QualificationError(
            f"review decision run_id is not UTC %Y%m%dT%H%M%SZ: {path}")
    try:
        datetime.strptime(run_id, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise QualificationError(
            f"review decision run_id is not a real UTC timestamp: "
            f"{run_id!r} ({path})"
        ) from exc
    for key in ("arm", "trigger"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise QualificationError(
                f"review decision {key} must be a non-empty string: {path}")
    if payload["decision"] not in REVIEW_DECISIONS:
        raise QualificationError(
            f"review decision must be one of {REVIEW_DECISIONS}: {path}")
    if not isinstance(payload["actor_attack_eligible"], bool):
        raise QualificationError(
            f"review decision actor_attack_eligible must be a boolean: {path}")
    if payload["decision"] == "exclude_actor_attack" and payload["actor_attack_eligible"]:
        raise QualificationError(
            f"review decision excludes ActorAttack but marks it eligible: {path}")
    if payload["decision"] != "exclude_actor_attack" and not payload["actor_attack_eligible"]:
        raise QualificationError(
            f"review decision keeps ActorAttack but marks it ineligible: {path}")
    for key in ("rationale", "decided_by"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise QualificationError(
                f"review decision {key} must be a non-empty string: {path}")
    decided_at = payload["decided_at"]
    if not isinstance(decided_at, str) or not decided_at.strip():
        raise QualificationError(
            f"review decision decided_at must be a non-empty ISO-8601 string: "
            f"{path}")
    try:
        datetime.fromisoformat(decided_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QualificationError(
            f"review decision decided_at is not valid ISO-8601: "
            f"{decided_at!r} ({path})"
        ) from exc
    return payload


def require_review_decision(
    report_block: dict[str, Any], path: str | Path | None = None
) -> dict[str, Any] | None:
    """Block an authorization when a triggered review has no valid decision.

    Phase 43 (P1-3): the decision is bound to the exact report SHA-256, run id,
    experiment configuration SHA-256 and attacker arm. A decision for another
    run is rejected, as is a decision supplied while no trigger is active.
    """
    triggers = report_block.get("review_triggers")
    if not isinstance(triggers, dict):
        raise QualificationError(
            "qualification report has no review_triggers mapping; refusing "
            "to infer that no review is required")
    required = sorted(
        name for name, block in triggers.items()
        if isinstance(block, dict) and block.get("triggered") is True
    )
    if not required:
        if path is not None:
            raise QualificationError(
                f"a review decision was supplied at {path}, but no review "
                f"trigger is active for this qualification report; refusing "
                f"to silently discard it.")
        return None
    if path is None:
        raise QualificationError(
            f"qualification report triggered a review for {required}; the "
            f"experiment is not authorized until a decision record is supplied "
            f"with --review-decision."
        )
    decision = load_review_decision(path)
    expected_config = (report_block.get("experiment_config") or {}).get("sha256")
    checks = {
        "qualification_report_sha256": report_block.get("sha256"),
        "run_id": report_block.get("run_id"),
        "configuration_sha256": expected_config,
        "arm": report_block.get("attacker_prompt_condition"),
    }
    mismatches = [
        f"{key} {decision.get(key)!r} != {value!r}"
        for key, value in checks.items()
        if decision.get(key) != value
    ]
    if mismatches:
        raise QualificationError(
            "review decision does not belong to this qualification report: "
            + "; ".join(mismatches)
            + f" ({path})"
        )
    if decision["trigger"] not in required:
        raise QualificationError(
            f"review decision names trigger {decision['trigger']!r}, but the "
            f"report triggered {required}: {path}"
        )
    return decision


# --------------------------------------------------------------------------- #
# Centralized R1 paid authorization (Phase 43, P0-1)
# --------------------------------------------------------------------------- #

def add_paid_authorization_args(parser) -> Any:
    """Add the two explicit authorization paths to a paid CLI parser."""
    parser.add_argument(
        "--qualification-report", default=None,
        help="Path to the QUALIFIED v2 report. Defaults to the newest "
             "results/qualification/<tag>.<arm>.*.qualification.v2.json.",
    )
    parser.add_argument(
        "--review-decision", default=None,
        help="JSON decision record required when the report triggered the "
             "delivery_type review.",
    )
    return parser


def is_r1_paid_model(model_id: Any, cfg: dict[str, Any] | None = None) -> bool:
    """True for the frozen R1 snapshot or a paper-reproduction configuration.

    The exact-snapshot rule is deliberate: ``gpt-4o`` and other aliases are not
    silently treated as the frozen R1 identity. Broadening this rule requires an
    explicit decision and tests.
    """
    return is_r1_model(model_id, cfg)


def _effective_runtime_config(
    cfg: dict[str, Any] | None,
    *,
    role: str | None = None,
    model_id: Any = None,
) -> dict[str, Any]:
    """Build the one effective runtime identity used by the gate.

    ``cfg=None`` retains the legacy exact-R1 contract: the frozen R1 config is
    the identity for an exact-R1 client. For a supplied config the role models
    must be explicit; a missing/empty role fails closed. When a caller has
    already selected the actual model for a role (a CLI override, a single-model
    role, or a role builder), that model replaces the configured role ID so the
    same value is authorized and constructed.
    """
    if cfg is None:
        effective = dict(load_config(R1_EXPERIMENT_CONFIG))
    else:
        effective = dict(cfg)
    effective_models = _effective_role_models(effective)
    actual = str(model_id or "").strip()
    if role is not None:
        if role not in QUALIFICATION_ROLES:
            raise PaidAuthorizationError(
                f"unknown paid role {role!r}; expected one of "
                f"{list(QUALIFICATION_ROLES)}")
        if actual:
            effective_models[role] = {
                **effective_models[role],
                "model": actual,
            }
    elif actual:
        known = {entry["model"] for entry in effective_models.values()}
        if actual not in known:
            raise PaidAuthorizationError(
                f"cannot determine which effective role would construct "
                f"{actual!r}; pass the role explicitly. Refusing to authorize "
                "a model that is not in the effective runtime configuration."
            )
    elif cfg is not None and effective.get("paper_reproduction"):
        raise PaidAuthorizationError(
            "paper-reproduction authorization requires a concrete role or "
            "model_id; refusing to authorize an unidentified runtime model.")
    effective["models"] = effective_models
    return effective


def require_verified_source(cfg: dict[str, Any] | None) -> None:
    """Require a loaded, re-hashable source file for a supplied config.

    ``_source_sha256`` is evidence only when accompanied by the source path the
    loader recorded. A hand-built mapping with no path, or with a claimed hash
    that does not match the file, fails closed. This is an application-policy
    check for trusted repository callers, not a hostile-process boundary.
    """
    if cfg is None:
        return
    path_value = cfg.get("_source_path")
    claimed = cfg.get("_source_sha256")
    if (not isinstance(path_value, str) or not path_value.strip()
            or not isinstance(claimed, str) or not claimed.strip()):
        raise PaidAuthorizationError(
            "paper-reproduction authorization requires a verified source file; "
            "load the configuration with the normal config loader (which records "
            "_source_path and _source_sha256). A hand-built config without a "
            "source identity is refused.")
    path = Path(path_value)
    if not path.is_file():
        raise PaidAuthorizationError(
            f"paper-reproduction source file is missing: {path}")
    actual = sha256_of(path)
    if actual != claimed.strip().lower():
        raise PaidAuthorizationError(
            f"paper-reproduction source file hash changed or does not match "
            f"the recorded identity: {path} (recorded {claimed[:12]}…, "
            f"actual {actual[:12]}…)")


def _r1_authorization_config(
    cfg: dict[str, Any] | None,
    *,
    role: str | None = None,
    model_id: Any = None,
) -> dict[str, Any]:
    effective = _effective_runtime_config(cfg, role=role, model_id=model_id)
    qualification = dict(effective.get("qualification") or {})
    qualification.setdefault(
        "output_dir", str(_PROJECT_ROOT / "results" / "qualification"))
    effective["qualification"] = qualification
    return effective


def require_paid_authorization(
    model_id: Any = None,
    *,
    cfg: dict[str, Any] | None = None,
    role: str | None = None,
    qualification_report: str | Path | None = None,
    review_decision: str | Path | None = None,
) -> dict[str, Any]:
    """Guard an R1 paid client construction.

    There is no caller-supplied exemption. The qualification harness uses the
    separate private builder ``_build_qualification_role_llm`` instead. Every
    other R1 purpose requires a current v2 report and, when triggered, a
    decision bound to that exact report. Models outside a paper-reproduction
    config that are not the exact R1 snapshot return unguarded.

    The report identity is derived from the effective runtime role models,
    including the actual role/model selected by the caller. A mismatch is
    rejected before the report is accepted or a client is constructed.
    """
    if not is_r1_paid_model(model_id, cfg):
        return {"gated": False, "model": model_id}
    if not str(model_id or "").strip():
        raise PaidAuthorizationError(
            "R1 paid authorization requires a non-empty effective model_id")
    if cfg is not None:
        require_verified_source(cfg)
    experiment_cfg = _r1_authorization_config(
        cfg, role=role, model_id=model_id)
    block = require_qualification(experiment_cfg, qualification_report)
    decision = require_review_decision(block, review_decision)
    return {
        "gated": True,
        "model": model_id,
        "effective_models": experiment_cfg.get("models"),
        "qualification": block,
        "review_decision": decision,
    }


def guard_paid_model(
    model_id: Any,
    args=None,
    *,
    cfg: dict[str, Any] | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for CLI scripts holding ``qualification_report``
    and ``review_decision`` on ``args``."""
    return require_paid_authorization(
        model_id,
        cfg=cfg,
        role=role,
        qualification_report=getattr(args, "qualification_report", None),
        review_decision=getattr(args, "review_decision", None),
    )


def build_paid_chat_llm(
    model_id: Any,
    *,
    args=None,
    cfg: dict[str, Any] | None = None,
    role: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
):
    """Gate and construct one direct OpenAI client (Phase 45/47/51).

    This is the supported path for scripts that previously called
    ``OpenAIChatLLM`` directly. It validates authorization before the client
    constructor (which reads the key) and opens the client-layer context.

    ``cfg`` contract: pass the paper-reproduction configuration whenever one is
    in use. For a supplied config, the effective three-role identity is derived
    from the actual config role models. ``cfg=None`` is only the non-paper/
    exact-model legacy path; the frozen R1 snapshot remains gated regardless.
    Pass ``role`` when the actual model is an override so the effective identity
    records which role it belongs to.
    """
    require_paid_authorization(
        model_id, cfg=cfg, role=role,
        qualification_report=getattr(args, "qualification_report", None),
        review_decision=getattr(args, "review_decision", None),
    )
    from guardbound.llm.openai_client import OpenAIChatLLM

    with _authorized(model_id):
        return OpenAIChatLLM(model_id=model_id, **(model_kwargs or {}))


def build_paid_role_llm(
    role_cfg: dict[str, Any],
    role: str,
    *,
    args=None,
    cfg: dict[str, Any] | None = None,
    manager=None,
    **kwargs,
):
    """Gate and build one role through the provider factory (Phase 45/47).

    Pass the paper-reproduction ``cfg`` whenever one is in use. ``cfg=None`` is
    only the non-paper/exact-model path; the frozen R1 snapshot is still gated
    at the client layer.
    """
    model_id = role_cfg.get("model")
    block = require_paid_authorization(
        model_id, cfg=cfg, role=role,
        qualification_report=getattr(args, "qualification_report", None),
        review_decision=getattr(args, "review_decision", None),
    )
    snapshot = dict(role_cfg)
    effective = block.get("effective_models") or {}
    if block.get("gated") and role in effective:
        snapshot["model"] = effective[role]["model"]
    from guardbound.llm.provider_factory import build_role_llm

    with _authorized(model_id):
        return build_role_llm(snapshot, role, manager=manager, **kwargs)


# --------------------------------------------------------------------------- #
# Ledger (observational telemetry)
# --------------------------------------------------------------------------- #

def build_ledger(
    cfg: dict[str, Any],
    *,
    rate_card_path: str | Path | None = None,
) -> UsageLedger:
    """Build the usage ledger.

    Observational only: Phase 22 removed the custom spend cap, so the ledger
    records tokens and cost and never blocks a request.
    """
    cloud_cfg = cfg.get("cloud") or {}
    card_path = rate_card_path or cloud_cfg.get("rate_card") or DEFAULT_RATE_CARD
    rate_card = load_rate_card(card_path)
    key = os.environ.get(str(cloud_cfg.get("api_key_env", "OPENAI_API_KEY")))
    return UsageLedger(
        rate_card=rate_card,
        secrets=(key,) if key else (),
    )


def pricing_warnings(ledger: UsageLedger, models: list[str]) -> list[str]:
    """Human-readable caveats about cost figures and model lifetime.

    Printed before a paid run so a deprecated model's closing reproducibility
    window is visible at the moment the spend is authorised, not discovered
    afterwards.
    """
    warnings: list[str] = []
    card = ledger.rate_card
    if not card.verified:
        warnings.append(
            f"rate card {card.version or '(unversioned)'} is UNVERIFIED "
            f"(as_of={card.as_of or 'unknown'}). Costs are estimates only; "
            f"confirm prices at {card.source or 'the provider pricing page'} "
            f"and set verified: true."
        )
    seen: set[str] = set()
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        rate = card.rate_for(model)
        if rate is None:
            warnings.append(
                f"no rate for {model!r}: its cost will be reported as null "
                f"rather than guessed."
            )
            continue
        if rate.shutdown_date:
            warnings.append(
                f"{model!r} is deprecated and scheduled to shut down on "
                f"{rate.shutdown_date}. Runs after that date are impossible; "
                f"every manifest records the execution date."
            )
    if ledger.unavailable_usage_calls:
        warnings.append(
            f"{ledger.unavailable_usage_calls} response(s) carried no usage "
            "object: their cost is unknown, so the reported total is a lower "
            "bound."
        )
    return warnings


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def build_role_llms(
    cfg: dict[str, Any],
    ledger: UsageLedger,
    *,
    qualification_report: str | Path | None = None,
    review_decision: str | Path | None = None,
):
    """Build attacker/target/evaluator from the config's ``models`` block.

    Roles resolve independently, so mixed local/cloud stacks work unchanged.
    Cloud-wide defaults are applied first, so a manifest cannot report a
    ``structured_output`` mode the clients were never given.

    Phase 45: an R1 stack is gated and the whole construction runs inside the
    client-layer authorization context. An all-offline-mock stack is exempt
    because it creates no paid client. There is no caller-supplied purpose flag.

    The attacker is wrapped for the configured prompt condition (see
    ``guardbound.llm.attacker_arms``); the baseline condition is a strict
    pass-through, so the paper path is untouched.
    Returns ``(manager, {"attacker": llm, "target": llm, "evaluator": llm})``.
    """
    from guardbound.llm.attacker_arms import build_armed_attacker
    from guardbound.llm.provider_factory import build_three_model_stack

    models_cfg = cfg.get("models") or {}
    offline_mock = all(
        isinstance(models_cfg.get(role, {}).get("client"), OfflineMockClient)
        for role in ("attacker", "target", "evaluator")
    )
    r1_models = [
        models_cfg.get(role, {}).get("model")
        for role in ("attacker", "target", "evaluator")
        if is_r1_paid_model(models_cfg.get(role, {}).get("model"), cfg)
    ]
    if r1_models and not offline_mock:
        block = require_paid_authorization(
            r1_models[0], cfg=cfg,
            qualification_report=qualification_report,
            review_decision=review_decision,
        )
    else:
        block = {}
    effective = block.get("effective_models") or {}
    models_snapshot = {}
    for role in ("attacker", "target", "evaluator"):
        role_snapshot = dict(models_cfg.get(role) or {})
        if role in effective:
            role_snapshot["model"] = effective[role]["model"]
        models_snapshot[role] = role_snapshot

    with _authorized(*r1_models):
        apply_cloud_defaults(models_snapshot, cfg.get("cloud") or {})
        manager, llms = build_three_model_stack(
            models_snapshot,
            device=(cfg.get("hardware") or {}).get("device", "cuda"),
            ledger=ledger,
        )
    llms["attacker"] = build_armed_attacker(llms["attacker"], attacker_arm_name(cfg))
    return manager, llms


def attacker_arm_name(cfg: dict[str, Any]) -> str:
    """The configured attacker prompt condition (defaults to the paper baseline)."""
    from guardbound.llm.attacker_arms import ARM_BASELINE

    return str(cfg.get("attacker_prompt_condition") or ARM_BASELINE)


def arm_manifest_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """Manifest block: which attacker condition ran, and the exact prompt hashes."""
    from guardbound.llm.attacker_arms import attacker_prompt_fingerprints

    return attacker_prompt_fingerprints(
        attacker_arm_name(cfg), prompt_variant_of(cfg)
    )


def diff_config_keys(a: Any, b: Any, prefix: str = "") -> list[str]:
    """Dotted paths whose values differ between two configs."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        out: list[str] = []
        for key in sorted(keys):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(path)
            else:
                out.extend(diff_config_keys(a[key], b[key], path))
        return out
    return [] if a == b else [prefix]


def assert_arm_comparability(
    base_cfg: dict[str, Any],
    arm_cfg: dict[str, Any],
    allowed_differences: tuple[str, ...] = (
        "experiment.name",
        "experiment.output_dir",
        "experiment.methodology_label",
        "attacker_prompt_condition",
    ),
) -> list[str]:
    """Refuse a comparison whose arms differ in anything but the prompt condition.

    Target model, evaluator, dataset, attacks, temperatures, NBF and seed must be
    identical, otherwise "the only difference is I1" is false.
    """
    differences = diff_config_keys(base_cfg, arm_cfg)
    unexpected = [d for d in differences if d not in allowed_differences]
    if unexpected:
        raise ValueError(
            "attacker-condition comparison is not controlled; configs differ at "
            f"{unexpected}"
        )
    return differences


def apply_cloud_defaults(
    models_cfg: dict[str, Any], cloud_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Push cloud-wide settings down into each cloud role, in place.

    A role's own key always wins. Without this, ``cloud.structured_output`` would
    be written into the manifest while the clients silently used their built-in
    default — a manifest that misreports the generation contract.
    """
    for role in ("attacker", "target", "evaluator"):
        role_cfg = models_cfg.get(role)
        if not isinstance(role_cfg, dict):
            continue
        if str(role_cfg.get("provider", "local")).lower() in (
            "local", "hf_local", "huggingface_local"
        ):
            continue
        for key in ("structured_output", "json_schema", "json_schema_name",
                    "timeout_s", "base_url", "organization", "project"):
            if key in cloud_cfg and cloud_cfg[key] is not None and key not in role_cfg:
                role_cfg[key] = cloud_cfg[key]
    return models_cfg


def _build_cloud_role(
    role: str, role_cfg: dict[str, Any], ledger: UsageLedger,
    attacker_arm: str | None,
):
    """Low-level cloud role construction. Callers must hold authorization."""
    from guardbound.llm.attacker_arms import build_armed_attacker
    from guardbound.llm.provider_factory import (
        build_cloud_role_llm,
        normalize_provider,
    )

    if normalize_provider(role_cfg.get("provider", "local")) != "cloud":
        raise ValueError(
            f"{role}: qualification only supports cloud roles; got "
            f"provider={role_cfg.get('provider')!r}"
        )
    llm = build_cloud_role_llm(role, role_cfg, ledger=ledger)
    return build_armed_attacker(llm, attacker_arm) if role == "attacker" else llm


def _qualification_capability(
    *, offline_mock: bool, confirmed: bool
) -> QualificationCapability:
    """Issue the internal qualification-harness capability.

    Private by convention; under the trusted in-process code model this is an
    internal capability, not a security primitive. A live capability requires
    ``confirmed=True``; an offline capability requires ``offline_mock=True``.
    """
    return _issue_qualification_capability(
        offline_mock=offline_mock, confirmed=confirmed)


def _build_qualification_role_llm(
    role: str, role_cfg: dict[str, Any], ledger: UsageLedger,
    attacker_arm: str | None = None,
    *,
    capability: QualificationCapability,
):
    """Qualification-harness-only client construction (Phase 47).

    This is the only path that may construct an R1 client without a
    qualification report, because the harness produces the first report. It
    requires an explicit ``QualificationCapability`` object issued by the
    harness's execution path; there is no filename, purpose string, or
    caller-controlled boolean on this function.

    * offline mode requires the role to carry an actual ``OfflineMockClient``;
    * live mode requires a confirmed capability (and therefore
      ``--confirm-paid-request`` at the harness);
    * construction runs in the owning execution context's authorization scope.
    """
    if not isinstance(capability, QualificationCapability):
        raise QualificationError(
            "a QualificationCapability issued by the qualification harness is "
            "required; a filename or purpose string does not grant the "
            "exemption."
        )
    if capability.offline_mock:
        if not isinstance(role_cfg.get("client"), OfflineMockClient):
            raise QualificationError(
                "offline qualification requires an actual OfflineMockClient; "
                f"got {type(role_cfg.get('client')).__name__}. Refusing to "
                f"construct a live provider or a non-mock client."
            )
    elif not capability.confirmed:
        raise QualificationError(
            "live qualification client construction requires "
            "--confirm-paid-request and a confirmed harness capability."
        )
    model_id = role_cfg.get("model")
    with _authorized(model_id):
        return _build_cloud_role(role, role_cfg, ledger, attacker_arm)


def build_single_role_llm(
    role: str, role_cfg: dict[str, Any], ledger: UsageLedger,
    attacker_arm: str | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    qualification_report: str | Path | None = None,
    review_decision: str | Path | None = None,
):
    """Build one cloud role through the research authorization gate.

    The qualification harness uses ``_build_qualification_role_llm`` instead;
    there is no caller-supplied purpose that disables this gate.

    Pass ``cfg`` for a paper-reproduction configuration; ``cfg=None`` is only
    the non-paper/exact-model path. The frozen R1 snapshot is always gated at
    the client layer.
    """
    model_id = role_cfg.get("model")
    block = require_paid_authorization(
        model_id, cfg=cfg, role=role,
        qualification_report=qualification_report,
        review_decision=review_decision,
    )
    snapshot = dict(role_cfg)
    effective = block.get("effective_models") or {}
    if block.get("gated") and role in effective:
        snapshot["model"] = effective[role]["model"]
    with _authorized(model_id):
        return _build_cloud_role(role, snapshot, ledger, attacker_arm)


# --------------------------------------------------------------------------- #
# Deterministic offline transport (harness validation only, never a result)
# --------------------------------------------------------------------------- #

class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.prompt_tokens_details = type("D", (), {"cached_tokens": 0})()
        self.completion_tokens_details = type("D", (), {"reasoning_tokens": 0})()


def fake_response(content: str, *, n_in: int = 100, n_out: int = 20):
    """A minimal 200 chat-completion object shaped like the SDK's response."""
    message = type("M", (), {"content": content, "refusal": None})()
    choice = type("C", (), {"message": message, "finish_reason": "stop"})()
    return type("R", (), {
        "choices": [choice],
        "usage": _FakeUsage(n_in, n_out),
        "id": "chatcmpl-offline-mock",
    })()


class OfflineMockClient:
    """Deterministic stand-in for ``openai.OpenAI``.

    Recognises the official call shapes by prompt content and answers with a
    syntactically valid reply, so the attack loop, the evaluator calls and the
    refusal/disclaimer checks all execute. It never touches the network, and its
    output is not a research measurement.
    """

    counter = 0

    def __init__(self, **kwargs) -> None:
        self.chat = type("Chat", (), {"completions": self})()
        self.seen_requests: list[dict] = []

    def create(self, **kwargs):
        self.seen_requests.append(kwargs)
        blob = "\n".join(
            str(m.get("content", "")) for m in kwargs.get("messages", [])
        )
        OfflineMockClient.counter += 1

        if "responseToEvaluateInput" in blob:
            if "disclaimer" in blob:
                payload = {"value": False, "rationale": "offline mock"}
            else:
                payload = {"value": False, "rationale": "offline mock", "metadata": 10}
        elif "scale of 1 to 5" in blob:
            payload = {"score": 1}
        elif "Extract the core harmful target" in blob:
            # ActorAttack pre-attack: extract_harm_target (official EXTRACT_PROMPT)
            payload = {
                "target": "offline mock harm target",
                "details": {"delivery_type": "article", "other_details": ""},
            }
        elif "choose most relevant" in blob:
            # ActorAttack pre-attack: get_actors (official ACTOR_PROMPT)
            payload = {"actors": [
                {"actor_name": f"offline mock actor {index}",
                 "relationship": "offline mock relationship"}
                for index in (1, 2, 3)
            ]}
        elif "you need extract the content of questions" in blob:
            # ActorAttack pre-attack: get_init_queries (JSON_FORMAT_QUESTION_PROMPT)
            payload = {"questions": [
                {"question": "offline mock actor question 1"},
                {"question": "offline mock actor question 2"},
            ]}
        elif "generatedQuestion" in blob:
            payload = {
                "generatedQuestion": f"offline mock question {OfflineMockClient.counter}",
                "lastResponseSummary": "offline mock summary",
            }
        else:
            payload = "offline mock target response"

        if kwargs.get("response_format", {}).get("type") in ("json_object", "json_schema"):
            return fake_response(json.dumps(payload))
        if isinstance(payload, str):
            return fake_response(payload)
        return fake_response(json.dumps(payload))


def resolver(role_cfg: dict[str, Any]) -> str:
    """A short ``provider:model`` description for manifests."""
    return f"{role_cfg.get('provider', 'local')}:{role_cfg.get('model')}"


# --------------------------------------------------------------------------- #
# Dataset / attacks
# --------------------------------------------------------------------------- #

def load_goals(cfg: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    dataset_cfg = cfg.get("dataset") or {}
    path = Path(dataset_cfg.get("path") or DEFAULT_DATASET)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. The paper's 200 HarmBench behaviours "
            f"ship with the official release."
        )
    with open(path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError(f"{path}: expected a JSON list of test cases")

    num_goals = dataset_cfg.get("num_goals")
    if num_goals is None:
        num_goals = len(cases)
    requested = int(num_goals)
    # Phase 38 (P2-6): fail closed rather than silently shrinking the ASR
    # denominator. A short file must abort before any client is constructed.
    if requested <= 0:
        raise ValueError(
            f"{path}: dataset.num_goals must be >= 1, got {requested}"
        )
    if len(cases) < requested:
        raise ValueError(
            f"{path}: dataset contains {len(cases)} behaviours but "
            f"dataset.num_goals requires {requested}. Refusing to start: the "
            f"paper's ASR denominator is the configured behaviour count and "
            f"must not shrink silently."
        )
    selected = cases[:requested]
    if limit is not None:
        requested_limit = int(limit)
        if requested_limit < 0:
            raise ValueError(f"--limit must be >= 0, got {requested_limit}")
        selected = selected[:requested_limit]
    for index, case in enumerate(selected):
        if not isinstance(case, dict) or not (case.get("task") or case.get("goal")):
            raise ValueError(
                f"{path}: selected row {index} is not a mapping with a "
                f"non-empty 'task' or 'goal' field (got "
                f"{type(case).__name__})"
            )
    return selected


#: Attacks whose attacker-generation calls accept ``attacker_temperature``.
#: Acronym is training-only and deliberately excluded (Phase 36 scope).
ATTACKER_TEMPERATURE_ATTACKS = frozenset({
    "actor_attack", "crescendo_paper", "opposite_day",
})


def build_attack(
    name: str,
    *,
    max_turns: int,
    attacker_llm,
    evaluator_llm,
    evaluator_temperature: float | None = None,
    attacker_temperature: float | None = None,
    prompt_variant: str | None = None,
):
    """Instantiate one paper attack with the evaluator backend wired in.

    The attack algorithms, prompts and JSON contracts are untouched; this only
    chooses which ``ChatLLM`` receives the attacker and evaluator calls, plus the
    paper-fidelity knobs (evaluator/attacker temperature and prompt-byte variant).
    """
    from guardbound.attacks.actor_attack import ActorAttack
    from guardbound.attacks.acronym import AcronymAttack
    from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
    from guardbound.attacks.opposite_day import OppositeDayAttack

    registry = {
        "actor_attack": ActorAttack,
        "crescendo_paper": CrescendoAttackPaper,
        "opposite_day": OppositeDayAttack,
        "acronym": AcronymAttack,
    }
    if name not in registry:
        raise ValueError(
            f"unsupported cloud attack {name!r}; supported: {sorted(registry)}"
        )
    knobs = {}
    if evaluator_temperature is not None:
        knobs["evaluator_temperature"] = float(evaluator_temperature)
    if attacker_temperature is not None and name in ATTACKER_TEMPERATURE_ATTACKS:
        knobs["attacker_temperature"] = float(attacker_temperature)
    if prompt_variant is not None:
        knobs["prompt_variant"] = prompt_variant
    attack = registry[name](max_turns=max_turns, attacker_llm=attacker_llm, **knobs)
    attack.set_evaluator_llm(evaluator_llm)
    return attack


def prompt_variant_of(cfg: dict[str, Any]) -> str:
    """The configured prompt-byte variant (``official`` for R1, ``legacy`` default)."""
    from guardbound.attacks.prompt_variants import (
        PROMPT_VARIANT_LEGACY,
        resolve_prompt_variant,
    )

    return resolve_prompt_variant(cfg.get("prompt_variant") or PROMPT_VARIANT_LEGACY)


def evaluator_temperature_of(cfg: dict[str, Any]) -> float:
    """The configured evaluator temperature (0.0 legacy default, 0.7 for R1)."""
    from guardbound.llm.base import LEGACY_EVALUATOR_TEMPERATURE

    temperature = cfg.get("temperature") or {}
    value = temperature.get("evaluator", LEGACY_EVALUATOR_TEMPERATURE)
    return float(LEGACY_EVALUATOR_TEMPERATURE if value is None else value)


def uses_composite_execution(cfg: dict[str, Any], attack_name: str) -> bool:
    """True when the configuration asks for ActorAttack's composite runner."""
    execution = (cfg.get("attacks") or {}).get("execution") or {}
    return str(execution.get(attack_name, "generic")).strip().lower() == "composite"


def validate_paper_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a paper-reproduction config; return its resolved settings.

    Non-paper configurations (qualification, historical cloud configs) are left
    alone — they are exploration harnesses, not the reproduction — but an unknown
    attack identifier is rejected everywhere.
    """
    from guardbound.paper_protocol import (
        resolve_paper_reproduction,
        validate_paper_attack_set,
    )

    validate_paper_attack_set(cfg)
    if not bool(cfg.get("paper_reproduction")):
        return None
    return resolve_paper_reproduction(cfg)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

# OpenAI snapshot ids end in a date (e.g. ``-2024-08-06``). Used only to label a
# configured id as pinned vs floating in a manifest — never to rewrite it.
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def execution_date() -> str:
    """UTC timestamp of the run, recorded because the models have a shutdown date."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def model_identity(
    models_cfg: dict[str, Any],
    declared: dict[str, Any] | None = None,
    *,
    temperature_cfg: dict[str, Any] | None = None,
    structured_output: str | None = None,
) -> dict[str, Any]:
    """Per-role model identity block for an experiment manifest.

    Records the configured id, whether it is a dated snapshot or a floating
    alias, and — when the provider does not expose one — states explicitly that
    no resolved revision is available. Every role is recorded separately, so an
    accidental shared id is visible rather than implied.
    """
    declared = declared or {}
    temperature_cfg = temperature_cfg or {}
    roles: dict[str, Any] = {}
    ids: list[str] = []

    for role in ("attacker", "target", "evaluator"):
        role_cfg = models_cfg.get(role) or {}
        model = role_cfg.get("model")
        ids.append(str(model))
        is_snapshot = bool(model) and bool(_SNAPSHOT_SUFFIX.search(str(model)))
        roles[role] = {
            "role": role,
            "configured_model_id": model,
            "provider": role_cfg.get("provider"),
            "endpoint": "/v1/chat/completions",
            "is_dated_snapshot": is_snapshot,
            "floating_alias": not is_snapshot,
            "snapshot_id": str(model) if is_snapshot else declared.get("snapshot"),
            "resolved_revision": None,
            "resolved_revision_note": (
                "The provider does not return the resolved model revision in the "
                "chat-completion response, so the served revision is not "
                "independently verifiable from the API result."
            ),
            "generation": {
                "temperature_used_by_calls": {
                    "attacker_generation": temperature_cfg.get("attacker"),
                    "target_completion": temperature_cfg.get("target"),
                    "evaluator_calls": temperature_cfg.get("evaluator"),
                },
                "structured_output_mode": structured_output,
                "max_output_tokens": role_cfg.get("max_output_tokens"),
            },
        }

    return {
        "execution_date_utc": execution_date(),
        "roles": roles,
        "all_roles_share_one_model": len({i for i in ids if i}) == 1,
        "deprecated": declared.get("deprecated"),
        "announced_shutdown_date": declared.get("announced_shutdown_date"),
        "identity_source": declared.get("source"),
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import importlib.metadata as md

        for package in ("openai", "torch", "sentence-transformers", "transformers"):
            try:
                info[package] = md.version(package)
            except Exception:  # noqa: BLE001 - optional dependency
                info[package] = "absent"
    except Exception:  # noqa: BLE001
        pass
    return info


def raw_threshold_from_eta(eta: float) -> float:
    """Raw admission threshold the runners expect for the paper's margin η.

    ``admit iff h < raw_threshold_from_eta(eta)`` is the paper's ``h < -eta``.
    This is the only place a configuration η is converted; passing η itself to a
    runner would invert the filter decision (Phase 23 defect D-01).
    """
    from guardbound.defense.filter_rule import threshold_from_eta

    return threshold_from_eta(float(eta))


def _git(*args: str) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        return out.stdout
    except Exception:  # noqa: BLE001 - git may be unavailable
        return None


def git_dirty() -> bool | None:
    """True when the working tree has uncommitted changes.

    A commit id alone does not identify the code that ran when the tree is dirty,
    which is the normal state between phases (Phase 23 defect D-06).
    """
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return bool(status.strip())


def git_status_sha256() -> str | None:
    """SHA-256 of ``git status --porcelain`` so a dirty tree is identifiable."""
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return hashlib.sha256(status.encode("utf-8")).hexdigest()


def paper_reference() -> dict[str, str]:
    """The paper this reproduction follows."""
    return {
        "id": "arXiv:2503.00187v3",
        "title": (
            "Steering Dialogue Dynamics for Robustness against Multi-turn "
            "Jailbreaking Attacks"
        ),
        "authors": "Hanjiang Hu, Alexander Robey, Changliu Liu",
        "venue": "Transactions on Machine Learning Research (TMLR), 2026-02",
    }


def dataset_provenance(
    cfg: dict[str, Any], goals: list[dict[str, Any]]
) -> dict[str, Any]:
    """Dataset identity, selection counts and goal-order hash for a manifest.

    ``num_goals_configured`` is the configuration's target count; the previous
    key (``num_goals_requested``) was the post-``--limit`` length and therefore
    misreported the plan (Phase 23 defect D-07).
    """
    dataset_cfg = cfg.get("dataset") or {}
    path = Path(dataset_cfg.get("path") or DEFAULT_DATASET)
    field = "task"
    ordered_ids = [str(case.get(field) or "") for case in goals]
    return {
        "name": "HarmBench behaviours (official NBF-LLM test split)",
        "path": str(path),
        "sha256": sha256_of(path) if path.exists() else None,
        "num_goals_configured": dataset_cfg.get("num_goals"),
        "num_goals_selected": len(goals),
        "goal_id_field": field,
        "goal_id_hash": hashlib.sha256(
            "\n".join(ordered_ids).encode("utf-8")
        ).hexdigest(),
        "sampling": dataset_cfg.get("sampling", "file_order"),
        "shuffle": bool(dataset_cfg.get("shuffle", False)),
    }


PAPER_ASR_DENOMINATOR_DEFINITION = (
    "successful behaviours / all configured behaviours per attack — the paper's "
    "definition (p8 Sec. 5.1: ASR is 'the ratio of successful jailbreaks ... over "
    "all harmful queries'). Infrastructure failures and invalid evaluator outcomes "
    "stay in the denominator and count as not successful."
)
VALID_ASR_DENOMINATOR_DEFINITION = (
    "successful behaviours / valid completed behaviours — diagnostic only "
    "(Phase 24 D-03). Infrastructure failures and invalid evaluator outcomes are "
    "excluded; this is never a substitute for the paper denominator."
)

PAPER_CODE_RETRY_POLICY: dict[str, Any] = retry_policy_record()



def prompt_provenance() -> dict[str, Any]:
    """Prompt-byte fingerprints for the manifest (Phase 29)."""
    try:
        from guardbound.attacks.prompt_variants import prompt_hashes

        return prompt_hashes()
    except Exception:  # noqa: BLE001 - a manifest must not fail on this
        return {}


def _ratio(numerator: int, denominator: int) -> float | None:
    """``None`` when the denominator is zero: ASR is undefined, not 0%."""
    return (numerator / denominator) if denominator else None


def aggregate_asr_report(
    rows: list[dict[str, Any]], expected_by_attack: dict[str, int]
) -> dict[str, Any]:
    """Dual ASR over per-goal outcome rows (Phase 29).

    ``rows`` are ``{"attack", "success", "valid_for_asr", "failed"}`` — one per
    planned goal. ``expected_by_attack`` is the planned goal count per attack,
    which is the paper's denominator.

    * a failed goal counts as *not successful* in the paper denominator and is
      excluded from the valid-completed one;
    * an invalid evaluator outcome counts as *not successful* in both;
    * neither is ever replaced by a successful or refusal verdict.
    """
    per_attack: dict[str, dict[str, Any]] = {}
    for name in expected_by_attack:
        per_attack[name] = {
            "attack": name,
            "expected_goals": int(expected_by_attack[name]),
            "completed_goals": 0,
            "valid_completed_goals": 0,
            "successful_goals": 0,
            "infrastructure_failure_count": 0,
            "invalid_evaluator_count": 0,
        }

    for row in rows:
        name = row.get("attack")
        if name not in per_attack:
            raise ValueError(
                f"ASR row for attack {name!r}, which is not in the plan "
                f"{sorted(per_attack)}"
            )
        bucket = per_attack[name]
        if row.get("failed"):
            bucket["infrastructure_failure_count"] += 1
            continue
        bucket["completed_goals"] += 1
        if not row.get("valid_for_asr"):
            bucket["invalid_evaluator_count"] += 1
            continue
        bucket["valid_completed_goals"] += 1
        if row.get("success") is True:
            bucket["successful_goals"] += 1

    for bucket in per_attack.values():
        bucket["asr_paper"] = _ratio(
            bucket["successful_goals"], bucket["expected_goals"])
        bucket["asr_valid_completed"] = _ratio(
            bucket["successful_goals"], bucket["valid_completed_goals"])

    totals = {
        key: sum(bucket[key] for bucket in per_attack.values())
        for key in (
            "expected_goals", "completed_goals", "valid_completed_goals",
            "successful_goals", "infrastructure_failure_count",
            "invalid_evaluator_count",
        )
    }
    totals["asr_paper"] = _ratio(
        totals["successful_goals"], totals["expected_goals"])
    totals["asr_valid_completed"] = _ratio(
        totals["successful_goals"], totals["valid_completed_goals"])
    return {
        "overall": totals,
        "by_attack": per_attack,
        "definitions": {
            "asr_paper": PAPER_ASR_DENOMINATOR_DEFINITION,
            "asr_valid_completed": VALID_ASR_DENOMINATOR_DEFINITION,
        },
    }


def actor_attack_prompt_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """Manifest block: identity + SHA-256 of every ActorAttack prompt byte string."""
    from guardbound.attacks.prompt_variants import actor_attack_prompt_provenance

    return actor_attack_prompt_provenance(prompt_variant_of(cfg))


UTILITY_READY = "READY FOR FUTURE AUTHORIZED EVALUATION"
UTILITY_BLOCKED_DATA = "BLOCKED — DATA REQUIRED"
UTILITY_BLOCKED_METHOD = "BLOCKED — METHODOLOGY UNRESOLVED"


def utility_manifest_block(
    cfg: dict[str, Any], root: str | Path | None = None
) -> dict[str, Any]:
    """Utility status/provenance for the manifest (Phase 30).

    The filesystem wins: a config can never *declare* a component ready while its
    dataset is absent or schema-invalid. Unresolved methodology is reported next
    to the data status, never hidden behind it.
    """
    from guardbound.evaluation.readiness import (
        STATUS_READY,
        upstream_provenance,
        utility_readiness,
    )

    declared = cfg.get("utility") or {}
    readiness = utility_readiness(root=root)
    components: dict[str, Any] = {}
    gaps: list[str] = []
    for name in ("mtbench", "xstest"):
        entry = dict(declared.get(name) or {})
        check = readiness["components"][name]
        entry_gaps = list(entry.pop("methodology_gaps", []) or [])
        declared_dataset = dict(entry.pop("dataset", {}) or {})
        # The declaration is advisory: the filesystem decides (Phase 30).
        declared_status = entry.pop("status", None)
        gaps.extend(entry_gaps)
        status = (UTILITY_READY if check["status"] == STATUS_READY
                  else UTILITY_BLOCKED_DATA)
        component = {
            "status": status,
            **entry,
            # Live facts win over the declaration: a config cannot claim rows or
            # a hash for a file that is not there.
            "dataset": {
                **declared_dataset,
                "path": check["path"],
                "exists": check["exists"],
                "rows": check["rows"],
                "missing_fields": check["missing_fields"],
                "sha256": (sha256_of(check["path"]) if check["exists"] else None),
            },
            "methodology_gaps": entry_gaps,
            # Immutable upstream pin (Phase 31): commit + git blob SHA-1 per file.
            # Recorded even while the dataset is absent, so an authorised future
            # run knows exactly which bytes to fetch.
            "upstream": upstream_provenance(name),
        }
        if check["status"] != STATUS_READY:
            component["blocked_reason"] = (
                "dataset absent or schema-invalid; no dataset is downloaded or "
                "substituted"
            )
        if declared_status and declared_status != status:
            component["declared_status"] = declared_status
        if entry_gaps:
            component["methodology_status"] = UTILITY_BLOCKED_METHOD
        components[name] = component

    table1_ready = all(
        component["status"] == UTILITY_READY for component in components.values()
    )
    return {
        "components": components,
        "methodology_gaps": gaps,
        "table1_ready": table1_ready,
        "status": UTILITY_READY if table1_ready else UTILITY_BLOCKED_DATA,
        "note": (
            "a working script or schema validator is not a completed utility "
            "evaluation; the utility columns stay blocked until the exact "
            "datasets and the unresolved methodology decisions are settled"
        ),
    }


def retry_summary_of(llm: Any) -> dict[str, Any] | None:
    """Retry telemetry for any backend that implements it (unwrapping arms)."""
    seen: set[int] = set()
    current = llm
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        summary = getattr(current, "retry_summary", None)
        if callable(summary):
            return summary()
        current = getattr(current, "_inner", None)
    return None


def retry_snapshot(llms: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-role cumulative retry counters (empty for backends without them)."""
    out: dict[str, dict[str, Any]] = {}
    for role, llm in (llms or {}).items():
        summary = retry_summary_of(llm)
        if summary is not None:
            out[role] = summary
    return out


_RETRY_COUNTERS = (
    "calls", "attempts", "retries", "scheduled_backoff_seconds",
    "elapsed_backoff_seconds", "successful_calls", "exhausted_calls",
    "non_retryable_calls",
)


def retry_delta(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Difference between two snapshots, for per-goal attribution."""
    roles: dict[str, Any] = {}
    for role, end in (after or {}).items():
        start = (before or {}).get(role) or {}
        delta = {
            key: end.get(key, 0) - start.get(key, 0) for key in _RETRY_COUNTERS
        }
        if not any(delta.values()):
            continue
        categories = {
            category: count - (start.get("by_category") or {}).get(category, 0)
            for category, count in (end.get("by_category") or {}).items()
        }
        delta["by_category"] = {
            category: count for category, count in categories.items() if count
        }
        # ``attempts`` is cumulative by construction; expose the same number under
        # the Phase 31 name so a per-goal row states its actual attempt count.
        delta["actual_attempts"] = delta["attempts"]
        roles[role] = delta
    if roles:
        roles["total_retries"] = sum(item.get("retries", 0) for item in roles.values())
    return roles


def nbf_manifest_block(
    nbf_cfg: dict[str, Any],
    nbf_meta: dict[str, Any] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """NBF provenance with the paper's terminology spelled out (Phase 30)."""
    eta = nbf_cfg.get("eta")
    block: dict[str, Any] = {
        "enabled": bool(nbf_cfg.get("enabled")),
        "eta": float(eta) if eta is not None else None,
        "raw_admission_threshold": -float(eta) if eta is not None else None,
        "filter_condition": "h + eta >= 0",
        "admission_condition": "h < -eta",
    }
    if nbf_meta:
        block.update(nbf_meta)
    if device:
        block["device"] = device
    block.setdefault("precision", "float32")
    return block


def git_commit() -> str | None:
    import subprocess


    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - git may be unavailable
        return None


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def append_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    return path


def safe(text: str) -> str:
    """Redact anything that looks like a credential before printing."""
    return redact(str(text), (os.environ.get("OPENAI_API_KEY") or "",))
