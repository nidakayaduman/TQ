"""Config-driven scenario scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config_loader import DEFAULT_APP_CONFIG, DEFAULT_SOFT_PENALTIES, load_scenario_weights, load_soft_penalties
from ..price_corridor import price_band_fit_score
from .scenario_validator import margin_pct

DEFAULT_SCORING_WEIGHTS = DEFAULT_APP_CONFIG["scenario_scoring"]


def _penalty_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name, {})
    return value if isinstance(value, dict) else {}


def _penalty_item(rule: str, penalty: float, message_tr: str, risk_flag: str) -> dict[str, Any]:
    return {
        "rule": rule,
        "penalty": float(penalty),
        "message_tr": message_tr,
        "message": message_tr,
        "risk_flag": risk_flag,
        "score_impact": -float(penalty),
    }


def _prediction_spread_ratio(scenario: dict[str, Any], corridor: dict[str, float]) -> float:
    predictions = scenario.get("baseline_predictions") or corridor.get("baseline_predictions") or []
    values: list[float] = []
    if isinstance(predictions, list):
        for item in predictions:
            if isinstance(item, dict) and item.get("prediction") is not None:
                values.append(float(item["prediction"]))
            elif isinstance(item, (int, float)):
                values.append(float(item))
    if not values:
        values = [
            float(corridor.get(key))
            for key in ["predicted_low_price", "predicted_mid_price", "predicted_high_price"]
            if corridor.get(key) is not None
        ]
    if len(values) < 2:
        return 0.0
    midpoint = max(float(np.median(values)), 1.0)
    return (max(values) - min(values)) / midpoint


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
    low_similarity = _penalty_section(cfg, "low_similarity")
    wide_band = _penalty_section(cfg, "wide_band")
    model_disagreement = _penalty_section(cfg, "model_disagreement")
    cluster_distance = _penalty_section(cfg, "cluster_distance")
    delivery_pressure = _penalty_section(cfg, "delivery_pressure")
    high_competition = _penalty_section(cfg, "high_competition")
    missing_optional_data = _penalty_section(cfg, "missing_optional_data")
    cost_uncertainty = _penalty_section(cfg, "cost_uncertainty")

    if proposed_price < corridor["predicted_low_price"] or proposed_price > corridor["predicted_high_price"]:
        items.append(
            _penalty_item(
                "price_outside_band",
                float(cfg["price_outside_band_penalty"]),
                "Fiyat koridoru dışında kalan teklif manuel kontrol gerektirir.",
                "outside_price_band",
            )
        )
    topk_similarity = float(profile_output.get("topk_avg_similarity", model_confidence_score / 100) or 0)
    if topk_similarity < float(low_similarity.get("threshold", 0.55)):
        items.append(
            _penalty_item(
                "low_similarity",
                float(low_similarity.get("penalty_points", cfg["low_similarity_penalty"])),
                "Bu senaryoda emsal ihale benzerliği düşük. Geçmiş kazanılmış ihalelerle yakınlık zayıf olduğu için model güveni azalır.",
                "low_similarity",
            )
        )
    if model_confidence_score < 45:
        items.append(
            _penalty_item(
                "low_model_confidence",
                float(cfg["low_model_confidence_penalty"]),
                "Model güveni düşük; emsal sinyali zayıf okunmalı.",
                "low_model_confidence",
            )
        )
    cluster_distance_value = 1 - (float(profile_output.get("cluster_score", 100) or 0) / 100)
    if (
        not bool(profile_output.get("is_inlier", True))
        or float(profile_output.get("won_profile_fit_score", 0)) < 45
        or cluster_distance_value > float(cluster_distance.get("threshold", 0.75))
    ):
        items.append(
            _penalty_item(
                "cluster_distance",
                float(cluster_distance.get("penalty_points", cfg["unusual_profile_penalty"])),
                "Bu ihale, geçmiş kazanılmış başarı grubunun tipik örneklerinden uzak görünüyor. Manuel inceleme önerilir.",
                "high_cluster_distance",
            )
        )
    if band_width_ratio > float(wide_band.get("normalized_band_width_threshold", 0.35)):
        items.append(
            _penalty_item(
                "wide_band",
                float(wide_band.get("penalty_points", cfg["wide_band_penalty"])),
                "Fiyat koridoru geniş. Gerçek fiyatı kapsama ihtimali artsa da karar desteği zayıflar.",
                "wide_price_band",
            )
        )
    spread_ratio = _prediction_spread_ratio(scenario, corridor)
    if spread_ratio > float(model_disagreement.get("prediction_spread_threshold_pct", 0.20)):
        items.append(
            _penalty_item(
                "model_disagreement",
                float(model_disagreement.get("penalty_points", cfg.get("model_disagreement_penalty", 6.0))),
                "Farklı fiyat modelleri arasında belirgin fark var. Bu nedenle fiyat önerisinin güveni düşürülür.",
                "medium_model_disagreement",
            )
        )
    delivery_days = int(scenario.get("delivery_days", int(scenario.get("delivery_months", tender.get("delivery_months", 0)) or 0) * 30))
    if delivery_days <= int(delivery_pressure.get("warning_days", 30)):
        items.append(
            _penalty_item(
                "delivery_pressure",
                float(delivery_pressure.get("penalty_points", cfg["high_delivery_penalty"])),
                "Teslimat süresi baskılı görünüyor. Operasyonel uygulanabilirlik ayrıca kontrol edilmelidir.",
                "delivery_pressure",
            )
        )
    if int(tender.get("competitor_count_estimate", 0) or 0) >= int(high_competition.get("competitor_count_threshold", 5)):
        items.append(
            _penalty_item(
                "high_competition",
                float(high_competition.get("penalty_points", cfg.get("competitor_pressure_penalty", 5.0))),
                "Tahmini rekabet seviyesi yüksek. Fiyat ve marj varsayımları daha dikkatli değerlendirilmelidir.",
                "high_competition",
            )
        )
    optional_fields = missing_optional_data.get(
        "optional_fields",
        ["competitor_count_estimate", "buyer_institution_type", "quantity_bucket", "delivery_bucket"],
    )
    missing = [field for field in optional_fields if tender.get(field) in (None, "")]
    if missing:
        penalty = min(
            float(missing_optional_data.get("max_penalty_points", cfg.get("data_completeness_penalty", 6.0))),
            len(missing) * float(missing_optional_data.get("penalty_per_missing_field", 1)),
        )
        items.append(
            _penalty_item(
                "missing_optional_data",
                penalty,
                "Bazı destekleyici veri alanları eksik. Bu durum model güvenini sınırlayabilir.",
                "missing_optional_data",
            )
        )
    uncertainty = str(tender.get("cost_uncertainty", scenario.get("cost_uncertainty", "low"))).casefold()
    threshold = str(cost_uncertainty.get("threshold", "medium")).casefold()
    if (threshold == "medium" and uncertainty in {"medium", "high"}) or (threshold == "high" and uncertainty == "high"):
        items.append(
            _penalty_item(
                "cost_uncertainty",
                float(cost_uncertainty.get("penalty_points", 5)),
                "Tahmini maliyet belirsiz görünüyor. Marj ve katkı kârı hesabı manuel olarak doğrulanmalıdır.",
                "cost_uncertainty",
            )
        )
    if validation.get("valid", False) and float(validation.get("computed_margin_pct", 0)) < 12:
        items.append(
            _penalty_item(
                "low_margin_buffer",
                float(cfg["low_margin_penalty"]),
                "Marj minimum eşiğe yakın; maliyet oynaklığına duyarlı.",
                "low_margin_buffer",
            )
        )
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
    risk_codes: list[str] = []
    if proposed_price < corridor["predicted_low_price"] or proposed_price > corridor["predicted_high_price"]:
        risk_flags.append("Fiyat tarihsel fiyat bandının dışında.")
        risk_codes.append("outside_price_band")
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
        risk_flag = str(item.get("risk_flag", ""))
        if risk_flag and risk_flag not in risk_codes:
            risk_codes.append(risk_flag)
    unusual_profile = not bool(profile_output.get("is_inlier", True)) or float(profile_output.get("won_profile_fit_score", 0)) < 45
    high_delivery = int(scenario.get("delivery_months", tender.get("delivery_months", 0)) or 0) > int(tender.get("delivery_months", 0) or 0)

    components = {
        "won_profile_fit_score": float(profile_output["won_profile_fit_score"]),
        "price_band_fit_score": price_band_fit_score(proposed_price, corridor),
        "margin_score": margin_score_from_pct(float(computed_margin)),
        "model_confidence_score": float(np.clip(model_confidence_score, 0, 100)),
        "risk_penalty_score": risk_penalty_score(
            risk_flags + risk_codes,
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
        "Senaryo geçerli. " + (f"Dikkat edilmesi gereken risk uyarısı: {soft_summary}" if soft_summary else "Belirgin risk uyarısı yok.")
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
        "caveat": "Kesin kural ihlali varsa ana öneri olarak kullanılmamalıdır." if not hard_valid else "Risk uyarısı varsa manuel kontrolle değerlendirilmelidir.",
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
                "anomaly_score",
                "isolation_threshold",
                "manual_review_flag",
                "manual_review_reasons",
                "topk_profile_score",
                "mixed_cluster_score",
                "cluster_purity_score",
                "profile_score_components",
                "cluster_id",
                "cluster_name",
                "cluster_score",
                "cluster_assignment_confidence",
                "cluster_distance",
                "cluster_second_distance",
                "cluster_distance_percentile",
                "cluster_count",
                "cluster_dominant_product_group",
                "cluster_dominant_product_group_ratio",
                "cluster_dominant_institution_type",
                "cluster_dominant_institution_type_ratio",
                "cluster_dominant_region",
                "cluster_dominant_region_ratio",
                "cluster_dominant_procedure_type",
                "cluster_dominant_procedure_type_ratio",
                "cluster_average_quantity",
                "cluster_median_delivery_months",
                "cluster_average_price",
                "cluster_average_margin",
                "cluster_median_price",
                "cluster_median_margin",
                "nearest_cluster_examples",
                "cluster_silhouette_score",
                "cluster_inertia",
                "cluster_min_size",
                "cluster_max_size",
                "small_cluster_count",
                "empty_cluster_count",
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
        "risk_codes": risk_codes,
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
            "risk_codes": risk_codes,
            "explainability": explainability,
        },
        **recommendation,
        "soft_penalty_score": components["risk_penalty_score"],
        "hard_constraints_valid": hard_valid,
    }
