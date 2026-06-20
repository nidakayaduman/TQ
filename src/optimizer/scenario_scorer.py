"""Config-driven scenario scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..price_corridor import price_band_fit_score
from .scenario_validator import margin_pct

DEFAULT_SCORING_WEIGHTS = {
    "won_profile_fit_score": 0.30,
    "price_band_fit_score": 0.25,
    "margin_score": 0.20,
    "model_confidence_score": 0.15,
    "risk_penalty_score": -0.10,
}


def margin_score_from_pct(value: float, target_margin_pct: float = 20.0) -> float:
    if value < 0:
        return 0.0
    return float(np.clip(value / max(target_margin_pct, 1.0) * 100, 0, 100))


def risk_penalty_score(flags: list[str], hard_constraint_valid: bool, low_confidence: bool = False) -> float:
    score = 0.0
    score += 45.0 if not hard_constraint_valid else 0.0
    score += min(35.0, len(flags) * 10.0)
    score += 15.0 if low_confidence else 0.0
    return float(np.clip(score, 0, 100))


def score_scenario(
    scenario: dict[str, Any],
    tender: dict[str, Any],
    corridor: dict[str, float],
    profile_output: dict[str, Any],
    model_confidence_score: float,
    validation: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_SCORING_WEIGHTS, **(weights or {})}
    proposed_price = float(scenario["proposed_unit_price"])
    unit_cost = float(scenario.get("estimated_unit_cost", tender.get("estimated_unit_cost", 0)))
    computed_margin = validation.get("computed_margin_pct", margin_pct(proposed_price, unit_cost))
    risk_flags = list(validation.get("violations", []))
    if proposed_price < corridor["predicted_low_price"] or proposed_price > corridor["predicted_high_price"]:
        risk_flags.append("Fiyat tarihsel fiyat bandının dışında.")

    components = {
        "won_profile_fit_score": float(profile_output["won_profile_fit_score"]),
        "price_band_fit_score": price_band_fit_score(proposed_price, corridor),
        "margin_score": margin_score_from_pct(float(computed_margin)),
        "model_confidence_score": float(np.clip(model_confidence_score, 0, 100)),
        "risk_penalty_score": risk_penalty_score(risk_flags, bool(validation.get("valid", False)), model_confidence_score < 45),
    }
    scenario_score = sum(cfg[key] * components[key] for key in cfg)
    return {
        **scenario,
        **{
            key: value
            for key, value in profile_output.items()
            if key
            in {
                "is_inlier",
                "inlier_score",
                "cluster_id",
                "cluster_name",
                "cluster_score",
                "cluster_count",
                "cluster_dominant_product_group",
                "cluster_dominant_institution_type",
                "cluster_dominant_region",
                "cluster_average_quantity",
                "cluster_average_price",
                "cluster_average_margin",
                "cluster_median_price",
                "cluster_median_margin",
                "isolation_contamination",
                "training_inlier_rate",
                "training_anomaly_rate",
                "segment_anomaly_rate",
            }
        },
        **components,
        "scenario_score": float(np.clip(scenario_score, 0, 100)),
        "computed_margin_pct": float(computed_margin),
        "risk_flags": risk_flags,
        "hard_constraints_valid": bool(validation.get("valid", False)),
    }
