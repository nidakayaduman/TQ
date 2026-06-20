"""Config-driven scenario scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config_loader import DEFAULT_APP_CONFIG, DEFAULT_SOFT_PENALTIES, load_scenario_weights, load_soft_penalties
from ..price_corridor import price_band_fit_score
from .scenario_validator import margin_pct

DEFAULT_SCORING_WEIGHTS = DEFAULT_APP_CONFIG["scenario_scoring"]


def margin_score_from_pct(value: float, target_margin_pct: float = 20.0) -> float:
    if value < 0:
        return 0.0
    return float(np.clip(value / max(target_margin_pct, 1.0) * 100, 0, 100))


def soft_penalty_items(
    scenario: dict[str, Any],
    tender: dict[str, Any],
    corridor: dict[str, float],
    profile_output: dict[str, Any],
    model_confidence_score: float,
    validation: dict[str, Any],
    penalties: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    cfg = {**DEFAULT_SOFT_PENALTIES, **(penalties or load_soft_penalties())}
    items: list[dict[str, Any]] = []
    proposed_price = float(scenario["proposed_unit_price"])
    mid = max(float(corridor.get("predicted_mid_price", proposed_price)), 0.01)
    band_width_ratio = float(corridor.get("band_width", 0)) / mid
    if proposed_price < corridor["predicted_low_price"] or proposed_price > corridor["predicted_high_price"]:
        items.append({"rule": "price_outside_band", "penalty": float(cfg["price_outside_band_penalty"]), "message": "Fiyat tarihsel fiyat bandının dışında."})
    if model_confidence_score < 45:
        items.append({"rule": "low_model_confidence", "penalty": float(cfg["low_model_confidence_penalty"]), "message": "Model güveni düşük; emsal sinyali zayıf okunmalı."})
    if not bool(profile_output.get("is_inlier", True)) or float(profile_output.get("won_profile_fit_score", 0)) < 45:
        items.append({"rule": "unusual_profile", "penalty": float(cfg["unusual_profile_penalty"]), "message": "Profil geçmiş kazanılmış örneklere göre sıra dışı görünüyor."})
    if band_width_ratio > 0.60:
        items.append({"rule": "wide_price_band", "penalty": float(cfg["wide_band_penalty"]), "message": "Fiyat bandı geniş; karar desteği daha temkinli okunmalı."})
    if int(tender.get("competitor_count_estimate", 0) or 0) >= 8:
        items.append({"rule": "competitor_pressure_high", "penalty": float(cfg["competitor_pressure_penalty"]), "message": "Tahmini rakip sayısı yüksek; rekabet baskısı artabilir."})
    required = ["product_group", "region", "procedure_type", "quantity", "delivery_months", "estimated_unit_cost"]
    missing = [field for field in required if tender.get(field) in (None, "")]
    if missing:
        items.append({"rule": "data_completeness_low", "penalty": float(cfg["data_completeness_penalty"]), "message": "Bazı zorunlu veri alanları eksik."})
    if validation.get("valid", False) and float(validation.get("computed_margin_pct", 0)) < 12:
        items.append({"rule": "low_margin_buffer", "penalty": float(cfg["low_margin_penalty"]), "message": "Marj minimum eşiğe yakın; maliyet oynaklığına duyarlı."})
    return items


def risk_penalty_score(
    flags: list[str],
    hard_constraint_valid: bool,
    low_confidence: bool = False,
    unusual_profile: bool = False,
    high_delivery: bool = False,
    penalties: dict[str, float] | None = None,
) -> float:
    cfg = {**DEFAULT_SOFT_PENALTIES, **(penalties or load_soft_penalties())}
    score = 0.0
    score += float(cfg["low_margin_penalty"]) if not hard_constraint_valid else 0.0
    score += min(35.0, len(flags) * float(cfg["high_risk_flag_penalty"]))
    score += float(cfg["insufficient_similar_count_penalty"]) if low_confidence else 0.0
    score += float(cfg.get("low_model_confidence_penalty", 0.0)) if low_confidence else 0.0
    score += float(cfg.get("unusual_profile_penalty", 0.0)) if unusual_profile else 0.0
    score += float(cfg.get("high_delivery_penalty", 0.0)) if high_delivery else 0.0
    if any("fiyat bandının dışında" in flag.casefold() for flag in flags):
        score += float(cfg["price_outside_band_penalty"])
    return float(np.clip(score, 0, 100))


def score_scenario(
    scenario: dict[str, Any],
    tender: dict[str, Any],
    corridor: dict[str, float],
    profile_output: dict[str, Any],
    model_confidence_score: float,
    validation: dict[str, Any],
    weights: dict[str, float] | None = None,
    soft_penalties: dict[str, float] | None = None,
) -> dict[str, Any]:
    cfg = {**load_scenario_weights(), **(weights or {})}
    proposed_price = float(scenario["proposed_unit_price"])
    unit_cost = float(scenario.get("estimated_unit_cost", tender.get("estimated_unit_cost", 0)))
    computed_margin = validation.get("computed_margin_pct", margin_pct(proposed_price, unit_cost))
    risk_flags = list(validation.get("violations", []))
    if proposed_price < corridor["predicted_low_price"] or proposed_price > corridor["predicted_high_price"]:
        risk_flags.append("Fiyat tarihsel fiyat bandının dışında.")
    detailed_soft_penalties = soft_penalty_items(
        scenario,
        tender,
        corridor,
        profile_output,
        model_confidence_score,
        validation,
        soft_penalties,
    )
    for item in detailed_soft_penalties:
        message = str(item["message"])
        if message not in risk_flags:
            risk_flags.append(message)
    unusual_profile = not bool(profile_output.get("is_inlier", True)) or float(profile_output.get("won_profile_fit_score", 0)) < 45
    high_delivery = int(scenario.get("delivery_months", tender.get("delivery_months", 0)) or 0) > int(tender.get("delivery_months", 0) or 0)

    components = {
        "won_profile_fit_score": float(profile_output["won_profile_fit_score"]),
        "price_band_fit_score": price_band_fit_score(proposed_price, corridor),
        "margin_score": margin_score_from_pct(float(computed_margin)),
        "model_confidence_score": float(np.clip(model_confidence_score, 0, 100)),
        "risk_penalty_score": risk_penalty_score(
            risk_flags,
            bool(validation.get("valid", False)),
            model_confidence_score < 45,
            unusual_profile,
            high_delivery,
            soft_penalties,
        ),
    }
    if detailed_soft_penalties:
        components["risk_penalty_score"] = float(
            np.clip(components["risk_penalty_score"] + sum(float(item["penalty"]) for item in detailed_soft_penalties) * 0.35, 0, 100)
        )
    scenario_score = sum(cfg[key] * components[key] for key in cfg)
    if not validation.get("valid", False) and not scenario.get("is_actual_configuration_candidate", False):
        scenario_score = 0.0
    hard_valid = bool(validation.get("valid", False))
    soft_summary = "; ".join(str(item["message"]) for item in detailed_soft_penalties)
    explainability = (
        "Senaryo geçerli. " + (f"Dikkat edilmesi gereken soft penalty: {soft_summary}" if soft_summary else "Belirgin soft penalty yok.")
        if hard_valid
        else "Senaryo geçersiz: " + "; ".join(validation.get("violations", []))
    )
    hard_status = "Uygun" if hard_valid else "Geçersiz"
    margin_impact = float(computed_margin)
    risk_impact = 100 - components["risk_penalty_score"]
    recommendation = {
        "changed_parameter": "unit_bid_price",
        "current_value": float(corridor.get("predicted_mid_price", proposed_price)),
        "recommended_value": proposed_price,
        "score_delta": float(np.clip(scenario_score, 0, 100) - 50),
        "margin_impact": margin_impact,
        "risk_impact": risk_impact,
        "confidence": float(np.clip(model_confidence_score, 0, 100)),
        "evidence_from_similar_tenders": "Benzer ihalelerin fiyat koridoru ve profil uyumu referans alındı.",
        "hard_constraint_status": hard_status,
        "caveat": "Hard constraint ihlali varsa ana öneri olarak kullanılmamalıdır." if not hard_valid else "Soft penalty varsa manuel kontrolle değerlendirilmelidir.",
    }
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
        "invalid_reason": "; ".join(validation.get("violations", [])) if not validation.get("valid", False) else "",
        "hard_constraint_violations": list(validation.get("violations", [])),
        "soft_penalties": detailed_soft_penalties,
        "soft_penalty_explanations": soft_summary,
        "is_valid": hard_valid,
        "explainability": explainability,
        "validator_output": {
            "scenario_id": scenario.get("scenario_id", ""),
            "is_valid": hard_valid,
            "hard_constraint_violations": list(validation.get("violations", [])),
            "soft_penalties": detailed_soft_penalties,
            "risk_flags": risk_flags,
            "explainability": explainability,
        },
        **recommendation,
        "soft_penalty_score": components["risk_penalty_score"],
        "hard_constraints_valid": hard_valid,
    }
