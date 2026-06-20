"""Synthetic stress scenarios for behavior checks."""

from __future__ import annotations

from typing import Any


def build_stress_scenarios(base_tender: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = []
    modifications = [
        ("extremely_high_quantity", {"quantity": float(base_tender.get("quantity", 1)) * 20}),
        ("extremely_low_quantity", {"quantity": 1}),
        ("product_buyer_mismatch", {"product_group": "Uyumsuz Ürün Grubu", "buyer_institution_type": "Bilinmeyen"}),
        ("very_short_delivery", {"delivery_months": 1}),
        ("very_long_delivery", {"delivery_months": 36}),
        ("high_competitor_count", {"competitor_count_estimate": 25}),
        ("low_similar_tender_match", {"product_name": "Bilinmeyen Molekül X"}),
    ]
    for name, patch in modifications:
        scenario = dict(base_tender)
        scenario.update(patch)
        scenario["stress_case"] = name
        scenarios.append(scenario)
    return scenarios

