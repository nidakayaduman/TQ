"""Scenario validation against hard constraints."""

from __future__ import annotations

from typing import Any


DEFAULT_HARD_CONSTRAINTS = {
    "minimum_margin_pct": 8.0,
    "minimum_delivery_months": 3,
    "max_deviation_above_historical_p90_pct": 30.0,
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


def margin_pct(unit_price: float, unit_cost: float) -> float:
    return ((unit_price - unit_cost) / unit_price * 100.0) if unit_price > 0 else -100.0


def validate_scenario(
    scenario: dict[str, Any],
    tender: dict[str, Any],
    corridor: dict[str, float],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_HARD_CONSTRAINTS, **(constraints or {})}
    violations: list[str] = []
    unit_price = float(scenario.get("proposed_unit_price", 0))
    unit_cost = float(scenario.get("estimated_unit_cost", tender.get("estimated_unit_cost", 0)))
    delivery_months = int(scenario.get("delivery_months", tender.get("delivery_months", 0)))
    computed_margin = margin_pct(unit_price, unit_cost)

    for field in cfg["required_fields"]:
        if tender.get(field) in (None, "") and scenario.get(field) in (None, ""):
            violations.append(f"Eksik zorunlu alan: {field}")

    if computed_margin < float(cfg["minimum_margin_pct"]):
        violations.append("Önerilen birim fiyat minimum marj eşiğinin altında.")
    if delivery_months < int(cfg["minimum_delivery_months"]):
        violations.append("Teslim süresi operasyonel minimumun altında.")

    p90 = float(corridor.get("p90", corridor.get("predicted_high_price", unit_price)))
    max_price = p90 * (1 + float(cfg["max_deviation_above_historical_p90_pct"]) / 100)
    if unit_price > max_price and not scenario.get("explicit_override", False):
        violations.append("Önerilen fiyat tarihsel p90 üst limitini aşıyor.")

    if scenario.get("product_alternative") and not scenario.get("product_alternative_allowed", False):
        violations.append("Ürün alternatifi şartname tarafından izinli değil.")

    return {
        "valid": not violations,
        "violations": violations,
        "computed_margin_pct": computed_margin,
    }

