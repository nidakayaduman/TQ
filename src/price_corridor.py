"""Historical won price corridors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import CANONICAL_PRICE_COLUMN


def percentile_corridor(prices: pd.Series, lower_q: float = 0.25, upper_q: float = 0.75) -> dict[str, float]:
    values = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if values.empty:
        raise ValueError("Price corridor requires at least one numeric price.")
    low = float(values.quantile(lower_q))
    mid = float(values.median())
    high = float(values.quantile(upper_q))
    return {
        "predicted_low_price": max(0.01, low),
        "predicted_mid_price": max(0.01, mid),
        "predicted_high_price": max(0.01, high),
        "p10": float(values.quantile(0.10)),
        "p25": low,
        "median": mid,
        "p75": high,
        "p90": float(values.quantile(0.90)),
        "average": float(values.mean()),
        "band_width": max(0.0, high - low),
    }


def corridor_for_similar_tenders(similar: pd.DataFrame, price_column: str = CANONICAL_PRICE_COLUMN) -> dict[str, float]:
    return percentile_corridor(similar[price_column])


def price_band_fit_score(proposed_price: float, corridor: dict[str, float]) -> float:
    low = corridor["predicted_low_price"]
    mid = corridor["predicted_mid_price"]
    high = corridor["predicted_high_price"]
    if low <= proposed_price <= high:
        distance = abs(proposed_price - mid) / max(high - low, 1.0)
        return float(np.clip(100 - distance * 35, 65, 100))
    nearest = low if proposed_price < low else high
    outside_ratio = abs(proposed_price - nearest) / max(mid, 1.0)
    return float(np.clip(65 - outside_ratio * 180, 0, 65))


def actual_inside_band(actual_price: float, corridor: dict[str, float]) -> bool:
    return bool(corridor["predicted_low_price"] <= actual_price <= corridor["predicted_high_price"])
