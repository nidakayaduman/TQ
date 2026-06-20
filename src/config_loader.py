"""Application configuration loading with safe defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"

DEFAULT_APP_CONFIG: dict[str, Any] = {
    "app": {
        "name": "Tender IQ Agentic Bid Advisor",
        "language": "tr",
        "default_top_k": 50,
    },
    "profile_fit": {
        "isolation_contamination": 0.05,
        "aggressive_anomaly_rate_threshold": 0.25,
    },
    "scenario_scoring": {
        "won_profile_fit_score": 0.30,
        "price_band_fit_score": 0.25,
        "margin_score": 0.20,
        "model_confidence_score": 0.15,
        "risk_penalty_score": -0.10,
    },
    "backtest": {
        "train_end_year": 2023,
        "validation_year": 2024,
        "test_year": 2025,
    },
}

DEFAULT_HARD_CONSTRAINTS: dict[str, Any] = {
    "margin": {
        "min_margin_pct": 0.08,
        "block_if_price_below_cost_plus_margin": True,
    },
    "price": {
        "max_price_over_topk_p90_multiplier": 1.15,
        "min_price_under_topk_p10_multiplier": 0.85,
        "require_override_for_outside_historical_band": True,
    },
    "delivery": {
        "operational_min_delivery_days": 15,
        "block_if_below_operational_minimum": True,
        "respect_mandatory_tender_deadline": True,
    },
    "product_alternative": {
        "require_spec_allows_alternative": True,
        "block_if_spec_flag_missing": True,
    },
    "data_quality": {
        "required_fields": [
            "tender_id",
            "tender_date",
            "product_name",
            "product_group",
            "buyer_institution",
            "region",
            "procedure_type",
            "quantity",
            "estimated_unit_cost",
        ],
        "block_if_required_fields_missing": True,
    },
    "leakage": {
        "blocked_fields_before_reveal": [
            "won_unit_price",
            "won_total_amount",
            "actual_margin_pct",
            "actual_unit_margin",
            "final_contract_amount",
            "actual_delivery_result",
            "actual_award_result",
        ],
        "fail_if_blocked_field_used": True,
    },
    "minimum_margin_pct": 8.0,
    "minimum_delivery_months": 3,
    "maximum_delivery_months": 36,
    "allow_product_alternative_default": False,
    "max_deviation_above_historical_p90_pct": 30.0,
    "max_price_over_p90_multiplier": 1.30,
    "min_price_under_p10_multiplier": 0.70,
    "minimum_quantity": 1,
    "minimum_unit_cost": 0.01,
    "required_fields": [
        "product_name",
        "product_group",
        "buyer_institution",
        "region",
        "procedure_type",
        "quantity",
        "delivery_months",
        "estimated_unit_cost",
    ],
}

DEFAULT_SOFT_PENALTIES: dict[str, Any] = {
    "low_similarity": {
        "threshold": 0.55,
        "penalty_points": 8,
    },
    "wide_band": {
        "normalized_band_width_threshold": 0.35,
        "penalty_points": 7,
    },
    "model_disagreement": {
        "prediction_spread_threshold_pct": 0.20,
        "penalty_points": 6,
    },
    "cluster_distance": {
        "threshold": 0.75,
        "penalty_points": 5,
    },
    "delivery_pressure": {
        "warning_days": 30,
        "penalty_points": 4,
    },
    "high_competition": {
        "competitor_count_threshold": 5,
        "penalty_points": 5,
    },
    "missing_optional_data": {
        "optional_fields": [
            "competitor_count_estimate",
            "buyer_institution_type",
            "quantity_bucket",
            "delivery_bucket",
        ],
        "penalty_per_missing_field": 1,
        "max_penalty_points": 6,
    },
    "cost_uncertainty": {
        "threshold": "medium",
        "penalty_points": 5,
    },
    "price_outside_band_penalty": 12.0,
    "low_similarity_penalty": 15.0,
    "unusual_quantity_penalty": 8.0,
    "low_margin_penalty": 12.0,
    "insufficient_similar_count_penalty": 15.0,
    "high_risk_flag_penalty": 10.0,
    "low_model_confidence_penalty": 10.0,
    "unusual_profile_penalty": 10.0,
    "high_delivery_penalty": 6.0,
}

DEFAULT_OBSERVABILITY_CONFIG: dict[str, Any] = {
    "logging": {
        "enabled": True,
        "level": "INFO",
        "directory": "logs",
        "application_log": "app.jsonl",
    },
    "audit": {
        "enabled": True,
        "directory": "audit_logs",
        "default_user_id": "anonymous",
    },
    "artifacts": {
        "enabled": True,
        "directory": "model_artifacts",
    },
}


def _deep_merge(default: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    merged = dict(default)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def load_app_config() -> dict[str, Any]:
    return _deep_merge(DEFAULT_APP_CONFIG, _read_yaml(CONFIG_DIR / "app_config.yaml"))


def _with_hard_constraint_aliases(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    margin = cfg.get("margin", {}) if isinstance(cfg.get("margin"), dict) else {}
    price = cfg.get("price", {}) if isinstance(cfg.get("price"), dict) else {}
    data_quality = cfg.get("data_quality", {}) if isinstance(cfg.get("data_quality"), dict) else {}

    min_margin_pct = float(margin.get("min_margin_pct", cfg.get("minimum_margin_pct", 8.0)))
    cfg["minimum_margin_pct"] = min_margin_pct * 100 if min_margin_pct <= 1 else min_margin_pct
    cfg["max_price_over_p90_multiplier"] = float(
        price.get("max_price_over_topk_p90_multiplier", cfg.get("max_price_over_p90_multiplier", 1.15))
    )
    cfg["min_price_under_p10_multiplier"] = float(
        price.get("min_price_under_topk_p10_multiplier", cfg.get("min_price_under_p10_multiplier", 0.85))
    )
    if "required_fields" not in cfg:
        cfg["required_fields"] = data_quality.get("required_fields", DEFAULT_HARD_CONSTRAINTS["required_fields"])
    return cfg


def _with_soft_penalty_aliases(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    alias_map = {
        "low_similarity_penalty": ("low_similarity", "penalty_points"),
        "wide_band_penalty": ("wide_band", "penalty_points"),
        "model_disagreement_penalty": ("model_disagreement", "penalty_points"),
        "unusual_profile_penalty": ("cluster_distance", "penalty_points"),
        "high_delivery_penalty": ("delivery_pressure", "penalty_points"),
        "competitor_pressure_penalty": ("high_competition", "penalty_points"),
        "data_completeness_penalty": ("missing_optional_data", "max_penalty_points"),
    }
    for alias, (section_name, key) in alias_map.items():
        section = cfg.get(section_name, {})
        if alias not in cfg and isinstance(section, dict):
            cfg[alias] = float(section.get(key, DEFAULT_SOFT_PENALTIES.get(alias, 0.0)))
    return cfg


def load_hard_constraints() -> dict[str, Any]:
    return _with_hard_constraint_aliases(_deep_merge(DEFAULT_HARD_CONSTRAINTS, _read_yaml(CONFIG_DIR / "hard_constraints.yaml")))


def load_soft_penalties() -> dict[str, Any]:
    loaded = _read_yaml(CONFIG_DIR / "soft_penalties.yaml")
    return _with_soft_penalty_aliases(_deep_merge(DEFAULT_SOFT_PENALTIES, loaded))


def load_observability_config() -> dict[str, Any]:
    return _deep_merge(DEFAULT_OBSERVABILITY_CONFIG, _read_yaml(CONFIG_DIR / "observability.yaml"))


def load_scenario_weights() -> dict[str, float]:
    config = load_app_config()
    weights = config.get("scenario_scoring", {})
    default = DEFAULT_APP_CONFIG["scenario_scoring"]
    merged = {**default, **weights}
    return {key: float(value) for key, value in merged.items()}


def active_config_summary() -> dict[str, Any]:
    app_config = load_app_config()
    return {
        "app_config": str(CONFIG_DIR / "app_config.yaml"),
        "hard_constraints": str(CONFIG_DIR / "hard_constraints.yaml"),
        "soft_penalties": str(CONFIG_DIR / "soft_penalties.yaml"),
        "default_top_k": int(app_config.get("app", {}).get("default_top_k", 50)),
        "isolation_contamination": float(app_config.get("profile_fit", {}).get("isolation_contamination", 0.05)),
        "aggressive_anomaly_rate_threshold": float(
            app_config.get("profile_fit", {}).get("aggressive_anomaly_rate_threshold", 0.25)
        ),
        "scenario_weights": load_scenario_weights(),
        "hard_constraint_keys": sorted(load_hard_constraints().keys()),
        "soft_penalty_keys": sorted(load_soft_penalties().keys()),
        "observability_config": str(CONFIG_DIR / "observability.yaml"),
    }
