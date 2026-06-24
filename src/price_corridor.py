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


def trend_adjusted_prices(
    similar: pd.DataFrame,
    price_column: str = CANONICAL_PRICE_COLUMN,
    target_year: int | None = None,
) -> pd.Series:
    frame = similar.copy()
    if "year" not in frame or "product_group" not in frame:
        return pd.to_numeric(frame[price_column], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
    valid = frame.dropna(subset=["year", price_column]).copy()
    if valid.empty or valid["year"].nunique() < 2:
        return pd.to_numeric(frame[price_column], errors="coerce")
    target = int(target_year if target_year is not None else valid["year"].max())
    global_year_medians = valid.groupby("year")[price_column].median().astype(float)
    adjusted = pd.Series(index=frame.index, dtype=float)
    for idx, row in frame.iterrows():
        price = row.get(price_column)
        year = row.get("year")
        if pd.isna(price) or pd.isna(year):
            adjusted.loc[idx] = price
            continue
        group = row.get("product_group")
        group_year_medians = valid[valid["product_group"] == group].groupby("year")[price_column].median().astype(float)
        trend_source = group_year_medians if len(group_year_medians) >= 3 else global_year_medians
        if len(trend_source) < 2:
            adjusted.loc[idx] = float(price)
            continue
        years = trend_source.index.astype(float).to_numpy()
        prices = trend_source.to_numpy(dtype=float)
        slope, intercept = np.polyfit(years, prices, 1)
        row_level = max(0.01, float(slope * float(year) + intercept))
        target_level = max(0.01, float(slope * target + intercept))
        factor = float(np.clip(target_level / row_level, 0.5, 1.5))
        adjusted.loc[idx] = max(0.01, float(price) * factor)
    return adjusted


def corridor_for_similar_tenders(similar: pd.DataFrame, price_column: str = CANONICAL_PRICE_COLUMN) -> dict[str, float]:
    return percentile_corridor(trend_adjusted_prices(similar, price_column))


def raw_corridor_for_similar_tenders(similar: pd.DataFrame, price_column: str = CANONICAL_PRICE_COLUMN) -> dict[str, float]:
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
