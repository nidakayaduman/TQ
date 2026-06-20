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
    "minimum_margin_pct": 8.0,
    "minimum_delivery_months": 3,
    "maximum_delivery_months": 36,
    "allow_product_alternative_default": False,
    "max_deviation_above_historical_p90_pct": 30.0,
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

DEFAULT_SOFT_PENALTIES: dict[str, float] = {
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


def load_hard_constraints() -> dict[str, Any]:
    return _deep_merge(DEFAULT_HARD_CONSTRAINTS, _read_yaml(CONFIG_DIR / "hard_constraints.yaml"))


def load_soft_penalties() -> dict[str, float]:
    loaded = _read_yaml(CONFIG_DIR / "soft_penalties.yaml")
    merged = {**DEFAULT_SOFT_PENALTIES, **loaded}
    return {key: float(value) for key, value in merged.items()}


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
    }
