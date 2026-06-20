"""Scenario validation against hard constraints."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config_loader import load_hard_constraints
from ..feature_masking import blocked_fields_present


def margin_pct(unit_price: float, unit_cost: float) -> float:
    return ((unit_price - unit_cost) / unit_price * 100.0) if unit_price > 0 else -100.0


def _missing(value: Any) -> bool:
    if value in (None, ""):
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name, {})
    return value if isinstance(value, dict) else {}


def _min_margin_fraction(cfg: dict[str, Any]) -> float:
    margin_cfg = _section(cfg, "margin")
    value = float(margin_cfg.get("min_margin_pct", cfg.get("minimum_margin_pct", 8.0)))
    return value if value <= 1 else value / 100


def _delivery_days(source: dict[str, Any], fallback_months: int) -> int:
    if source.get("delivery_days") not in (None, ""):
        return int(source.get("delivery_days"))
    if source.get("delivery_months") not in (None, ""):
        return int(source.get("delivery_months")) * 30
    return int(fallback_months) * 30


def validate_scenario(
    scenario: dict[str, Any],
    tender: dict[str, Any],
    corridor: dict[str, float],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_hard_constraints()
    if constraints:
        cfg = {**cfg, **constraints}
    margin_cfg = _section(cfg, "margin")
    price_cfg = _section(cfg, "price")
    delivery_cfg = _section(cfg, "delivery")
    product_cfg = _section(cfg, "product_alternative")
    data_quality_cfg = _section(cfg, "data_quality")
    leakage_cfg = _section(cfg, "leakage")

    violations: list[str] = []
    unit_price = float(scenario.get("proposed_unit_price", 0))
    unit_cost = float(scenario.get("estimated_unit_cost", tender.get("estimated_unit_cost", 0)))
    delivery_months = int(scenario.get("delivery_months", tender.get("delivery_months", 0)))
    quantity = float(scenario.get("quantity", tender.get("quantity", 0)))
    computed_margin = margin_pct(unit_price, unit_cost)

    required_fields = data_quality_cfg.get("required_fields", cfg.get("required_fields", []))
    missing_required = [field for field in required_fields if _missing(tender.get(field)) and _missing(scenario.get(field))]
    if missing_required and bool(data_quality_cfg.get("block_if_required_fields_missing", True)):
        violations.append(
            "Zorunlu veri alanları eksik olduğu için senaryo güvenilir şekilde değerlendirilemedi. "
            f"Eksik alanlar: {', '.join(missing_required)}"
        )

    min_margin_fraction = _min_margin_fraction(cfg)
    min_margin_price = unit_cost * (1 + min_margin_fraction)
    if bool(margin_cfg.get("block_if_price_below_cost_plus_margin", True)) and unit_price < min_margin_price:
        violations.append("Önerilen fiyat, tahmini maliyet ve minimum marj eşiğinin altında kaldığı için bu senaryo önerilemez.")
    if unit_price <= 0:
        violations.append("Önerilen fiyat sıfır veya negatif olamaz.")
    if unit_cost < float(cfg["minimum_unit_cost"]):
        violations.append("Tahmini birim maliyet mantıklı alt sınırın altında.")
    if unit_price < unit_cost:
        violations.append("Önerilen fiyat, tahmini maliyetin altında.")
    if quantity < float(cfg["minimum_quantity"]):
        violations.append("Miktar mantıklı alt sınırın altında.")
    if computed_margin < min_margin_fraction * 100:
        violations.append("Minimum marj eşiğinin altında kaldı.")
    if computed_margin < 0:
        violations.append("Negatif marj oluştu.")
    delivery_days = _delivery_days(scenario, delivery_months)
    operational_min_days = int(delivery_cfg.get("operational_min_delivery_days", int(cfg["minimum_delivery_months"]) * 30))
    if bool(delivery_cfg.get("block_if_below_operational_minimum", True)) and delivery_days < operational_min_days:
        violations.append("Teslim planı operasyonel minimum teslim süresinin altında kaldığı için uygulanabilir görünmüyor.")
    mandatory_deadline_days = tender.get("mandatory_delivery_days", tender.get("tender_deadline_delivery_days"))
    if (
        bool(delivery_cfg.get("respect_mandatory_tender_deadline", True))
        and mandatory_deadline_days not in (None, "")
        and delivery_days > int(mandatory_deadline_days)
    ):
        violations.append("Teslim planı ihale dokümanındaki zorunlu teslim tarihine uymuyor.")
    if delivery_months > int(cfg["maximum_delivery_months"]):
        violations.append("Teslim süresi belirlenen üst sınırı aşıyor.")

    p90 = float(corridor.get("p90", corridor.get("predicted_high_price", unit_price)))
    p10 = float(corridor.get("p10", corridor.get("predicted_low_price", unit_price)))
    max_multiplier = float(price_cfg.get("max_price_over_topk_p90_multiplier", cfg.get("max_price_over_p90_multiplier", 1.15)))
    max_price = p90 * max_multiplier
    override_required = bool(price_cfg.get("require_override_for_outside_historical_band", True))
    if unit_price > max_price and (override_required and not scenario.get("explicit_override", False)):
        violations.append(
            "Önerilen fiyat, benzer geçmiş ihalelerdeki üst fiyat seviyesinin belirgin şekilde üzerine çıkıyor. "
            "Manuel onay olmadan ana öneri yapılmamalıdır."
        )
    min_price = p10 * float(price_cfg.get("min_price_under_topk_p10_multiplier", cfg.get("min_price_under_p10_multiplier", 0.85)))
    if unit_price < min_price and (override_required and not scenario.get("explicit_override", False)):
        violations.append(
            "Önerilen fiyat, benzer geçmiş ihalelerdeki alt fiyat seviyesinin çok altında kalıyor. "
            "Bu durum marj ve sürdürülebilirlik açısından risklidir."
        )

    if scenario.get("product_alternative"):
        spec_flag = scenario.get("product_alternative_allowed", tender.get("product_alternative_allowed"))
        if _missing(spec_flag) and bool(product_cfg.get("block_if_spec_flag_missing", True)):
            violations.append("Şartname ürün alternatifi iznini net belirtmediği için alternatif ürün senaryosu önerilemez.")
        elif bool(product_cfg.get("require_spec_allows_alternative", True)) and not bool(spec_flag):
            violations.append("Şartname ürün alternatifi iznini net belirtmediği için alternatif ürün senaryosu önerilemez.")

    leakage_fields = blocked_fields_present([*tender.keys(), *scenario.keys()])
    configured_blocked = set(leakage_cfg.get("blocked_fields_before_reveal", []))
    leakage_fields = [field for field in leakage_fields if not configured_blocked or field in configured_blocked]
    if leakage_fields and bool(leakage_cfg.get("fail_if_blocked_field_used", True)):
        violations.append("Sızıntı tespit edildi. Gerçek sonuç alanları reveal öncesi senaryo değerlendirmesine giremez.")

    explainability = (
        "Senaryo geçerli. Kesin kural ihlali yok; risk uyarıları ayrıca skorlamada değerlendirilir."
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
