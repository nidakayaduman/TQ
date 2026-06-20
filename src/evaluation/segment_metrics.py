"""Segment-level evaluation."""

from __future__ import annotations

import pandas as pd


def segment_level_metrics(results: pd.DataFrame, segment_columns: list[str] | None = None) -> pd.DataFrame:
    columns = segment_columns or [
        "product_group",
        "region",
        "buyer_institution_type",
        "quantity_bucket",
        "cluster_name",
        "year",
        "cluster_id",
    ]
    rows: list[dict[str, object]] = []
    for column in [col for col in columns if col in results.columns]:
        for value, group in results.groupby(column, dropna=False):
            actual = group["actual_won_unit_price"].astype(float)
            mid = group["predicted_mid_price"].astype(float)
            errors = (actual - mid).abs()
            band_width = (group["predicted_high_price"] - group["predicted_low_price"]).astype(float)
            rows.append(
                {
                    "segment_column": column,
                    "segment_value": value,
                    "tender_count": int(len(group)),
                    "average_similarity": float(group["top50_avg_similarity"].mean()) if "top50_avg_similarity" in group else 0.0,
                    "price_corridor_coverage_rate": float(group["actual_inside_band"].mean()),
                    "mae": float(errors.mean()),
                    "mape": float((errors / actual.replace(0, pd.NA).abs() * 100).mean()),
                    "average_band_width": float(band_width.mean()),
                    "coverage_adjusted_band_score": float(group["coverage_adjusted_band_score"].mean())
                    if "coverage_adjusted_band_score" in group
                    else 0.0,
                    "anomaly_rate": float((~group["is_inlier"].astype(bool)).mean()) if "is_inlier" in group else 0.0,
                    "average_profile_fit": float(group["won_profile_fit_score"].mean()),
                    "low_confidence_rate": float((group["model_confidence_score"] < 50).mean()),
                }
            )
    return pd.DataFrame(rows)
