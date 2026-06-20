"""Backtest metrics for won-tender profile and price-band evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def smape(actual: pd.Series, predicted: pd.Series) -> float:
    a = actual.astype(float).to_numpy()
    p = predicted.astype(float).to_numpy()
    denominator = (np.abs(a) + np.abs(p)) / 2
    return float(np.mean(np.where(denominator == 0, 0, np.abs(a - p) / denominator)) * 100)


def wape(actual: pd.Series, predicted: pd.Series) -> float:
    a = actual.astype(float).to_numpy()
    p = predicted.astype(float).to_numpy()
    return float(np.sum(np.abs(a - p)) / max(np.sum(np.abs(a)), 1.0) * 100)


def price_corridor_metrics(results: pd.DataFrame) -> dict[str, float]:
    actual = results["actual_won_unit_price"].astype(float)
    mid = results["predicted_mid_price"].astype(float)
    errors = (actual - mid).abs()
    pct_errors = errors / actual.replace(0, np.nan).abs() * 100
    coverage_rate = float(results["actual_inside_band"].mean()) if len(results) else 0.0
    band_width = (results["predicted_high_price"] - results["predicted_low_price"]).astype(float)
    normalized_width = (band_width / mid.replace(0, np.nan).abs()).replace([np.inf, -np.inf], np.nan).fillna(0)
    normalized_band_width_penalty = float(np.clip(normalized_width.mean(), 0, 1))
    return {
        "mae": float(errors.mean()) if len(errors) else 0.0,
        "mape": float(pct_errors.mean()) if len(pct_errors) else 0.0,
        "smape": smape(actual, mid) if len(results) else 0.0,
        "wape": wape(actual, mid) if len(results) else 0.0,
        "median_absolute_error": float(errors.median()) if len(errors) else 0.0,
        "coverage_rate": coverage_rate,
        "band_coverage": coverage_rate,
        "average_band_width": float(band_width.mean()) if len(results) else 0.0,
        "normalized_band_width_penalty": normalized_band_width_penalty,
        "coverage_adjusted_band_score": float(coverage_rate * (1 - normalized_band_width_penalty)),
    }


def optimizer_metrics(results: pd.DataFrame) -> dict[str, float]:
    if results.empty:
        return {
            "actual_won_scenario_rank_percentile_mean": 0.0,
            "top30_hit_rate": 0.0,
            "top15_hit_rate": 0.0,
            "hard_constraint_violation_rate": 0.0,
        }
    percentile = results["actual_won_scenario_rank_percentile"].astype(float)
    return {
        "actual_won_scenario_rank_percentile_mean": float(percentile.mean()),
        "top30_hit_rate": float((percentile >= 70).mean()),
        "top15_hit_rate": float((percentile >= 85).mean()),
        "hard_constraint_violation_rate": float((~results["hard_constraints_valid"].astype(bool)).mean()),
    }
