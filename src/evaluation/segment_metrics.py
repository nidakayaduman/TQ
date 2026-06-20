"""Segment-level evaluation."""

from __future__ import annotations

import pandas as pd


def segment_level_metrics(results: pd.DataFrame, segment_columns: list[str] | None = None) -> pd.DataFrame:
    columns = segment_columns or [
        "product_group",
        "buyer_institution_type",
        "region",
        "procedure_type",
        "quantity_bucket",
        "delivery_bucket",
        "year",
        "cluster_id",
    ]
    rows: list[dict[str, object]] = []
    for column in [col for col in columns if col in results.columns]:
        for value, group in results.groupby(column, dropna=False):
            actual = group["actual_won_unit_price"].astype(float)
            mid = group["predicted_mid_price"].astype(float)
            errors = (actual - mid).abs()
            rows.append(
                {
                    "segment_column": column,
                    "segment_value": value,
                    "tender_count": int(len(group)),
                    "mae": float(errors.mean()),
                    "mape": float((errors / actual.replace(0, pd.NA).abs() * 100).mean()),
                    "coverage": float(group["actual_inside_band"].mean()),
                    "average_profile_fit": float(group["won_profile_fit_score"].mean()),
                    "low_confidence_rate": float((group["model_confidence_score"] < 50).mean()),
                }
            )
    return pd.DataFrame(rows)

