"""Synthetic stress scenarios for behavior checks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..advisor.fallback_advisor import build_fallback_advisor
from ..config_loader import load_scenario_weights, load_soft_penalties
from ..optimizer.recommendation_engine import rank_scenarios
from ..optimizer.scenario_scorer import score_scenario
from ..optimizer.scenario_validator import validate_scenario


def build_stress_scenarios(base_tender: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = []
    modifications = [
        ("extremely_high_quantity", {"quantity": float(base_tender.get("quantity", 1)) * 20}),
        ("extremely_low_quantity", {"quantity": 1}),
        ("product_buyer_mismatch", {"product_group": "Uyumsuz Ürün Grubu", "buyer_institution_type": "Bilinmeyen"}),
        ("very_short_delivery", {"delivery_months": 1}),
        ("very_long_delivery", {"delivery_months": 36}),
        ("very_high_price", {"stress_price_mode": "very_high"}),
        ("very_low_price", {"stress_price_mode": "very_low"}),
        ("high_competitor_count", {"competitor_count_estimate": 25}),
        ("low_similar_tender_match", {"product_name": "Bilinmeyen Molekül X"}),
    ]
    for name, patch in modifications:
        scenario = dict(base_tender)
        scenario.update(patch)
        scenario["stress_case"] = name
        scenarios.append(scenario)
    return scenarios


def _case_label(case: str) -> str:
    labels = {
        "extremely_high_quantity": "Aşırı yüksek miktar",
        "extremely_low_quantity": "Aşırı düşük miktar",
        "product_buyer_mismatch": "Ürün / kurum uyumsuzluğu",
        "very_short_delivery": "Çok kısa teslim süresi",
        "very_long_delivery": "Çok uzun teslim süresi",
        "very_high_price": "Çok yüksek fiyat",
        "very_low_price": "Çok düşük fiyat",
        "high_competitor_count": "Çok yüksek rakip varsayımı",
        "low_similar_tender_match": "Düşük emsal benzerliği",
    }
    return labels.get(case, case)


def evaluate_synthetic_outliers(train_df: pd.DataFrame, base_tender: dict[str, Any], top_k: int = 50) -> pd.DataFrame:
    """Evaluate whether synthetic edge cases trigger lower confidence or more review signals."""
    train = train_df.copy()
    base_result = rank_scenarios(train, base_tender, top_k=top_k)
    base_best = base_result["scenarios"].iloc[0].to_dict()
    base_profile = float(base_best.get("won_profile_fit_score", 0))
    base_confidence = float(base_result.get("model_confidence_score", 0))
    rows: list[dict[str, Any]] = []
    weights = load_scenario_weights()
    soft_penalties = load_soft_penalties()

    for stress_tender in build_stress_scenarios(base_tender):
        stress_case = str(stress_tender.pop("stress_case"))
        price_mode = stress_tender.pop("stress_price_mode", "")
        result = rank_scenarios(train, stress_tender, top_k=top_k)
        best = result["scenarios"].iloc[0].to_dict()

        if price_mode:
            corridor = result["corridor"]
            unit_cost = float(stress_tender.get("estimated_unit_cost", 0.01) or 0.01)
            proposed_price = (
                float(corridor["predicted_high_price"]) * 2.5
                if price_mode == "very_high"
                else max(0.01, unit_cost * 0.55)
            )
            manual_scenario = {
                "scenario_id": f"STRESS_{stress_case}",
                "proposed_unit_price": proposed_price,
                "estimated_unit_cost": unit_cost,
                "delivery_months": int(stress_tender.get("delivery_months", 0) or 0),
                "price_anchor": float(corridor["predicted_mid_price"]),
                "price_multiplier": 1.0,
                "is_actual_configuration_candidate": False,
            }
            validation = validate_scenario(manual_scenario, stress_tender, corridor)
            best = score_scenario(
                manual_scenario,
                stress_tender,
                corridor,
                best,
                float(result.get("model_confidence_score", 0)),
                validation,
                weights=weights,
                soft_penalties=soft_penalties,
            )

        advisor = build_fallback_advisor({**best, "similar_tender_count": len(result["similar"])})
        profile_score = float(best.get("won_profile_fit_score", 0))
        confidence_score = float(result.get("model_confidence_score", 0))
        risk_flags = list(best.get("risk_flags", []))
        triggered = (
            profile_score < base_profile
            or confidence_score < base_confidence
            or bool(risk_flags)
            or bool(advisor.get("manual_review_required"))
        )
        rows.append(
            {
                "stress_case": stress_case,
                "Senaryo": _case_label(stress_case),
                "Profil uyumu": profile_score,
                "Model güveni": confidence_score,
                "Risk bayrağı sayısı": len(risk_flags),
                "Manuel inceleme önerisi": "Evet" if advisor.get("manual_review_required") else "Hayır",
                "Beklenen davranış": "Geçti" if triggered else "İncelenmeli",
                "Risk notları": "; ".join(risk_flags) if risk_flags else "Belirgin risk bayrağı yok",
            }
        )
    return pd.DataFrame(rows)
