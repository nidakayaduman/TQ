"""Candidate scenario generation."""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_candidate_scenarios(
    tender: dict[str, Any],
    corridor: dict[str, float],
    include_actual: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    anchors = [
        corridor["predicted_low_price"],
        corridor["predicted_mid_price"],
        corridor["predicted_high_price"],
    ]
    multipliers = [0.95, 1.0, 1.05]
    unit_cost = float(tender.get("estimated_unit_cost", tender.get("estimated_unit_cost_try", 0)))
    scenarios: list[dict[str, Any]] = []
    for anchor in anchors:
        for multiplier in multipliers:
            price = float(max(0.01, anchor * multiplier))
            scenarios.append(
                {
                    "scenario_id": f"S{len(scenarios)+1:03d}",
                    "proposed_unit_price": round(price, 4),
                    "estimated_unit_cost": unit_cost,
                    "delivery_months": int(tender.get("delivery_months", 0)),
                    "price_anchor": round(float(anchor), 4),
                    "price_multiplier": multiplier,
                    "is_actual_configuration_candidate": False,
                }
            )
    if include_actual:
        actual_price = float(include_actual["actual_won_unit_price"])
        scenarios.append(
            {
                "scenario_id": "S_ACTUAL_HISTORICAL_CONFIG",
                "proposed_unit_price": round(actual_price, 4),
                "estimated_unit_cost": unit_cost,
                "delivery_months": int(tender.get("delivery_months", 0)),
                "price_anchor": round(float(np.median(anchors)), 4),
                "price_multiplier": 1.0,
                "is_actual_configuration_candidate": True,
            }
        )
    return scenarios

