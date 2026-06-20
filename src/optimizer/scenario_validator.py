"""Scenario validation against hard constraints."""

from __future__ import annotations

from typing import Any

from ..config_loader import load_hard_constraints


def margin_pct(unit_price: float, unit_cost: float) -> float:
    return ((unit_price - unit_cost) / unit_price * 100.0) if unit_price > 0 else -100.0


def validate_scenario(
    scenario: dict[str, Any],
    tender: dict[str, Any],
    corridor: dict[str, float],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**load_hard_constraints(), **(constraints or {})}
    violations: list[str] = []
    unit_price = float(scenario.get("proposed_unit_price", 0))
    unit_cost = float(scenario.get("estimated_unit_cost", tender.get("estimated_unit_cost", 0)))
    delivery_months = int(scenario.get("delivery_months", tender.get("delivery_months", 0)))
    quantity = float(scenario.get("quantity", tender.get("quantity", 0)))
    computed_margin = margin_pct(unit_price, unit_cost)

    for field in cfg["required_fields"]:
        if tender.get(field) in (None, "") and scenario.get(field) in (None, ""):
            violations.append(f"Eksik zorunlu alan: {field}")

    min_margin_price = unit_cost * (1 + float(cfg["minimum_margin_pct"]) / 100)
    if unit_price < min_margin_price:
        violations.append("Önerilen fiyat, tahmini maliyetin ve minimum marj eşiğinin altında kalamaz.")
    if unit_price <= 0:
        violations.append("Önerilen fiyat sıfır veya negatif olamaz.")
    if unit_cost < float(cfg["minimum_unit_cost"]):
        violations.append("Tahmini birim maliyet mantıklı alt sınırın altında.")
    if unit_price < unit_cost:
        violations.append("Önerilen fiyat, tahmini maliyetin altında.")
    if quantity < float(cfg["minimum_quantity"]):
        violations.append("Miktar mantıklı alt sınırın altında.")
    if computed_margin < float(cfg["minimum_margin_pct"]):
        violations.append("Minimum marj eşiğinin altında kaldı.")
    if computed_margin < 0:
        violations.append("Negatif marj oluştu.")
    if delivery_months < int(cfg["minimum_delivery_months"]):
        violations.append("Teslim süresi operasyonel minimumun altında.")
    if delivery_months > int(cfg["maximum_delivery_months"]):
        violations.append("Teslim süresi belirlenen üst sınırı aşıyor.")

    p90 = float(corridor.get("p90", corridor.get("predicted_high_price", unit_price)))
    p10 = float(corridor.get("p10", corridor.get("predicted_low_price", unit_price)))
    max_multiplier = float(cfg.get("max_price_over_p90_multiplier", 1 + float(cfg["max_deviation_above_historical_p90_pct"]) / 100))
    max_price = p90 * max_multiplier
    if unit_price > max_price and not scenario.get("explicit_override", False):
        violations.append("Fiyat, geçmiş emsal koridorunun çok dışında.")
    min_price = p10 * float(cfg.get("min_price_under_p10_multiplier", 0.70))
    if unit_price < min_price and not scenario.get("explicit_override", False):
        violations.append("Önerilen fiyat, benzer geçmiş ihalelerdeki alt fiyat seviyesinin çok altına düşemez.")

    if scenario.get("product_alternative") and not scenario.get("product_alternative_allowed", False):
        violations.append("Ürün alternatifi şartname tarafından izinli değil.")

    explainability = (
        "Senaryo geçerli. Hard constraint ihlali yok; soft penalty ve risk sinyalleri ayrıca skorlamada değerlendirilir."
        if not violations
        else "Senaryo geçersiz: " + "; ".join(violations)
    )
    return {
        "scenario_id": scenario.get("scenario_id", ""),
        "valid": not violations,
        "is_valid": not violations,
        "violations": violations,
        "hard_constraint_violations": violations,
        "soft_penalties": [],
        "risk_flags": violations,
        "explainability": explainability,
        "computed_margin_pct": computed_margin,
    }
