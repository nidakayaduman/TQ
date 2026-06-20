"""Data quality checks."""

from __future__ import annotations

import pandas as pd

from .constants import CANONICAL_MARGIN_COLUMN, CANONICAL_PRICE_COLUMN
from .schema import normalize_schema


def validate_data_quality(df: pd.DataFrame) -> dict[str, object]:
    data = normalize_schema(df)
    issues: list[str] = []

    if data.empty:
        issues.append("Veri seti boş.")
    if "tender_id" in data and data["tender_id"].duplicated().any():
        issues.append("Tekrarlı tender_id kayıtları var.")

    numeric_checks = {
        "quantity": "Miktar sıfırdan büyük olmalı.",
        "delivery_months": "Teslim süresi sıfırdan büyük olmalı.",
        "competitor_count_estimate": "Tahmini rakip sayısı negatif olmamalı.",
        CANONICAL_PRICE_COLUMN: "Kazanılmış birim fiyat sıfırdan büyük olmalı.",
    }
    for column, message in numeric_checks.items():
        if column in data.columns:
            values = pd.to_numeric(data[column], errors="coerce")
            if values.isna().any():
                issues.append(f"{column} sayısal olmayan değer içeriyor.")
            threshold = 0 if column != "competitor_count_estimate" else -1
            if (values <= threshold).any():
                issues.append(message)

    if CANONICAL_MARGIN_COLUMN in data.columns:
        margins = pd.to_numeric(data[CANONICAL_MARGIN_COLUMN], errors="coerce")
        if ((margins < -100) | (margins > 100)).any():
            issues.append("Marj yüzdesi beklenen aralığın dışında.")

    return {
        "passed": not issues,
        "issues": issues,
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
    }

