"""Pricing verification and model-snapshot pinning.

Locks in the two Phase-3/Phase-4 guarantees:

* the shipped rate card is internally consistent, verified, and resolves dated
  snapshots to their family rate;
* every runnable cloud config pins a dated snapshot rather than a floating
  alias, and manifests record the identity plus a limitation note.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

from guardbound.llm.usage import (
    COST_BASIS_RATE_CARD,
    ModelRate,
    load_rate_card,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATE_CARD = PROJECT_ROOT / "configs" / "cloud_pricing.yaml"
SNAPSHOT_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _scripts_common():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import _cloud_common  # noqa: PLC0415

    return _cloud_common


# --------------------------------------------------------------------------- #
# Rate card
# --------------------------------------------------------------------------- #

def test_rate_card_is_marked_verified_and_dated():
    card = load_rate_card(RATE_CARD)
    assert card.verified is True
    assert card.cost_basis == COST_BASIS_RATE_CARD
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", card.as_of)
    assert card.source.startswith("https://")


def test_official_prices_match_the_verified_source():
    """Values transcribed from the official pricing table on 2026-11-10."""
    card = load_rate_card(RATE_CARD)

    gpt4o = card.rate_for("gpt-4o")
    assert (gpt4o.input_per_1m, gpt4o.cached_input_per_1m, gpt4o.output_per_1m) == (
        2.50, 1.25, 10.00)

    mini = card.rate_for("gpt-4o-mini")
    assert (mini.input_per_1m, mini.cached_input_per_1m, mini.output_per_1m) == (
        0.15, 0.075, 0.60)


def test_dated_snapshots_resolve_to_their_family_rate():
    card = load_rate_card(RATE_CARD)
    assert card.rate_for("gpt-4o-2024-08-06").input_per_1m == 2.50
    assert card.rate_for("gpt-4o-mini-2024-07-18").output_per_1m == 0.60
    # An unpinned alias still resolves to the family rate.
    assert card.rate_for("gpt-4o-mini").input_per_1m == 0.15


def test_shutdown_dates_are_carried_on_the_rate():
    card = load_rate_card(RATE_CARD)
    assert card.rate_for("gpt-4o-mini").shutdown_date == "2027-01-14"
    assert card.rate_for("gpt-4o").shutdown_date == "2027-01-14"


def test_shutdown_model_produces_a_warning_before_a_paid_run():
    common = _scripts_common()
    from guardbound.llm.usage import UsageLedger

    ledger = UsageLedger(rate_card=load_rate_card(RATE_CARD))
    warnings = common.pricing_warnings(ledger, ["gpt-4o-mini-2024-07-18"])
    assert any("shut down on 2027-01-14" in w for w in warnings)


def test_shut_down_models_are_not_priced():
    """A price for a shut-down model would silently mis-cost a run."""
    card = load_rate_card(RATE_CARD)
    for gone in ("gpt-3.5-turbo-0125", "o1-2024-12-17"):
        assert card.rate_for(gone) is None, f"{gone} must not carry a stale price"


def test_unpriced_model_still_returns_null_cost_not_zero():
    from guardbound.llm.usage import estimate_cost

    assert estimate_cost(ModelRate("unknown"), input_tokens=100, output_tokens=10) is None


def test_cost_arithmetic_on_a_deterministic_sample_usage_record():
    """600 uncached + 400 cached input, 500 output, at the gpt-4o-mini rates."""
    card = load_rate_card(RATE_CARD)
    rate = card.rate_for("gpt-4o-mini-2024-07-18")
    expected = 600 * 0.15 / 1e6 + 400 * 0.075 / 1e6 + 500 * 0.60 / 1e6
    from guardbound.llm.usage import estimate_cost

    got = estimate_cost(
        rate, input_tokens=1000, output_tokens=500, cached_tokens=400)
    assert got == pytest.approx(expected)
    # And the same record through the ledger summary.
    assert got == pytest.approx(0.00042, abs=1e-9)


def test_gpt4o_cost_arithmetic():
    from guardbound.llm.usage import estimate_cost

    card = load_rate_card(RATE_CARD)
    rate = card.rate_for("gpt-4o-2024-08-06")
    # 1000 in, 400 cached, 500 out
    expected = 600 * 2.50 / 1e6 + 400 * 1.25 / 1e6 + 500 * 10.00 / 1e6
    assert estimate_cost(
        rate, input_tokens=1000, output_tokens=500, cached_tokens=400
    ) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Config snapshot pinning
# --------------------------------------------------------------------------- #

CLOUD_CONFIGS = (
    "configs/cloud_research_gpt4o_mini.yaml",
    "configs/cloud_research_reference.yaml",
    "configs/cloud_qualification.yaml",
)


@pytest.mark.parametrize("config_name", CLOUD_CONFIGS)
def test_runnable_cloud_configs_pin_dated_snapshots(config_name):
    cfg = yaml.safe_load((PROJECT_ROOT / config_name).read_text(encoding="utf-8"))
    for role in ("attacker", "target", "evaluator"):
        model = cfg["models"][role]["model"]
        assert SNAPSHOT_RE.search(model), (
            f"{config_name}: models.{role}.model={model!r} is a floating alias; "
            f"pin a dated snapshot"
        )


@pytest.mark.parametrize("config_name", CLOUD_CONFIGS)
def test_pinned_snapshot_has_a_verified_rate(config_name):
    cfg = yaml.safe_load((PROJECT_ROOT / config_name).read_text(encoding="utf-8"))
    card = load_rate_card(RATE_CARD)
    for role in ("attacker", "target", "evaluator"):
        model = cfg["models"][role]["model"]
        assert card.rate_for(model) is not None, f"no rate for {model}"


def test_configuration_b_uses_the_mini_snapshot_and_labels_the_substitution():
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/cloud_research_gpt4o_mini.yaml").read_text(
            encoding="utf-8"))
    models = {cfg["models"][r]["model"] for r in ("attacker", "target", "evaluator")}
    assert models == {"gpt-4o-mini-2024-07-18"}
    assert cfg["model_identity"]["snapshot"] == "gpt-4o-mini-2024-07-18"
    assert cfg["model_identity"]["deprecated"] is True
    assert cfg["model_identity"]["announced_shutdown_date"] == "2027-01-14"
    assert "SUBSTITUTION" in cfg["experiment"]["methodology_label"].upper()


def test_configuration_a_uses_the_paper_snapshot():
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/cloud_research_reference.yaml").read_text(
            encoding="utf-8"))
    models = {cfg["models"][r]["model"] for r in ("attacker", "target", "evaluator")}
    assert models == {"gpt-4o-2024-08-06"}


def test_smoke_test_default_model_is_a_snapshot():
    import cloud_smoke_test  # noqa: PLC0415

    parser_default = cloud_smoke_test.parse_args([]).model
    assert SNAPSHOT_RE.search(parser_default)


# --------------------------------------------------------------------------- #
# Manifest model identity
# --------------------------------------------------------------------------- #

def test_model_identity_records_each_role_separately():
    common = _scripts_common()
    models_cfg = {
        "attacker": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
        "target": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
        "evaluator": {"provider": "openai", "model": "gpt-4o-2024-08-06"},
    }
    identity = common.model_identity(
        models_cfg,
        {"deprecated": True, "announced_shutdown_date": "2027-01-14",
         "source": "https://example.invalid"},
        temperature_cfg={"attacker": 0.7, "target": 0.7, "evaluator": 0.0},
        structured_output="json_object",
    )
    assert set(identity["roles"]) == {"attacker", "target", "evaluator"}
    assert identity["roles"]["attacker"]["configured_model_id"] == "gpt-4o-mini-2024-07-18"
    assert identity["roles"]["evaluator"]["configured_model_id"] == "gpt-4o-2024-08-06"
    assert identity["all_roles_share_one_model"] is False
    assert identity["deprecated"] is True
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                        identity["execution_date_utc"])


def test_model_identity_flags_a_floating_alias_and_states_the_limitation():
    common = _scripts_common()
    identity = common.model_identity(
        {"attacker": {"provider": "openai", "model": "gpt-4o-mini"},
         "target": {"provider": "openai", "model": "gpt-4o-mini"},
         "evaluator": {"provider": "openai", "model": "gpt-4o-mini"}},
    )
    attacker = identity["roles"]["attacker"]
    assert attacker["is_dated_snapshot"] is False
    assert attacker["floating_alias"] is True
    assert attacker["resolved_revision"] is None
    assert "not" in attacker["resolved_revision_note"].lower()
    assert identity["all_roles_share_one_model"] is True


def test_model_identity_records_generation_parameters_per_role():
    common = _scripts_common()
    identity = common.model_identity(
        {"attacker": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
         "target": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
         "evaluator": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"}},
        temperature_cfg={"attacker": 0.7, "target": 0.7, "evaluator": 0.0},
        structured_output="json_object",
    )
    temps = identity["roles"]["evaluator"]["generation"]["temperature_used_by_calls"]
    assert temps == {
        "attacker_generation": 0.7,
        "target_completion": 0.7,
        "evaluator_calls": 0.0,
    }
    assert (identity["roles"]["attacker"]["generation"]["structured_output_mode"]
            == "json_object")


def test_cloud_defaults_reach_the_clients_so_the_manifest_cannot_lie():
    """`cloud.structured_output` must be applied, not merely reported."""
    common = _scripts_common()
    models = {
        "attacker": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
        "target": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
        "evaluator": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
    }
    common.apply_cloud_defaults(
        models, {"structured_output": "json_schema", "timeout_s": 90})
    for role in ("attacker", "target", "evaluator"):
        assert models[role]["structured_output"] == "json_schema"
        assert models[role]["timeout_s"] == 90


def test_role_level_setting_wins_over_the_cloud_default():
    common = _scripts_common()
    models = {
        "attacker": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18",
                     "structured_output": "json_object"},
        "target": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
        "evaluator": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
    }
    common.apply_cloud_defaults(models, {"structured_output": "json_schema"})
    assert models["attacker"]["structured_output"] == "json_object"
    assert models["target"]["structured_output"] == "json_schema"


def test_cloud_defaults_do_not_touch_local_roles():
    common = _scripts_common()
    models = {
        "attacker": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
        "target": {"provider": "local", "model": "microsoft/phi-4"},
        "evaluator": {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
    }
    common.apply_cloud_defaults(models, {"structured_output": "json_schema"})
    assert "structured_output" not in models["target"]


def test_pilot_manifest_includes_model_identity(monkeypatch, tmp_path, capsys):
    """A harness-validation run must still record identity, snapshot and date."""
    import json

    import cloud_pilot  # noqa: PLC0415

    out = tmp_path / "pilot"
    code = cloud_pilot.main([
        "--offline-mock",
        "--variant", "cloud_baseline_a_nbf_off",
        "--attack", "crescendo_paper",
        "--limit", "1",
        "--out-dir", str(out),
    ])
    assert code == 0, capsys.readouterr().err
    manifest = json.loads(
        (out / "cloud_baseline_a_nbf_off.manifest.json").read_text(encoding="utf-8"))
    identity = manifest["model_identity"]
    assert identity["roles"]["attacker"]["is_dated_snapshot"] is True
    assert identity["roles"]["attacker"]["configured_model_id"] == (
        "gpt-4o-mini-2024-07-18")
    assert identity["announced_shutdown_date"] == "2027-01-14"
    assert identity["execution_date_utc"]
