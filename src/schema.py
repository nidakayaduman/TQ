"""Schema normalization and validation for won tender data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    CANONICAL_MARGIN_COLUMN,
    CANONICAL_PRICE_COLUMN,
    CANONICAL_TOTAL_COLUMN,
    REQUIRED_BASE_COLUMNS,
)


PRICE_ALIASES = [
    "won_unit_price",
    "inflation_adjusted_unit_price_2026_try",
    "winning_unit_price_try",
]
TOTAL_ALIASES = ["won_total_amount", "inflation_adjusted_contract_value_2026_try", "contract_value_try"]
MARGIN_ALIASES = ["actual_margin_pct", "gross_margin_pct"]
COST_ALIASES = ["estimated_unit_cost", "estimated_unit_cost_try", "internal_unit_cost_try"]


@dataclass(frozen=True)
class SchemaValidationResult:
    valid: bool
    missing_columns: list[str]
    warnings: list[str]
    row_count: int
    columns: list[str]


def _first_existing(columns: pd.Index, aliases: list[str]) -> str | None:
    return next((column for column in aliases if column in columns), None)


def add_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "quantity_bucket" not in normalized.columns:
        values = pd.to_numeric(normalized["quantity"], errors="coerce")
        labels = ["Düşük", "Orta", "Yüksek"]
        try:
            normalized["quantity_bucket"] = pd.qcut(values.rank(method="first"), q=3, labels=labels)
        except ValueError:
            normalized["quantity_bucket"] = "Orta"
        normalized["quantity_bucket"] = normalized["quantity_bucket"].astype(str)

    if "delivery_bucket" not in normalized.columns:
        delivery = pd.to_numeric(normalized["delivery_months"], errors="coerce")
        normalized["delivery_bucket"] = np.select(
            [delivery <= 3, delivery >= 12],
            ["Kısa", "Uzun"],
            default="Orta",
        )
    return normalized


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map current MVP columns into canonical won-tender fields."""

    normalized = df.copy()
    if "tender_date" in normalized.columns:
        normalized["tender_date"] = pd.to_datetime(normalized["tender_date"], errors="coerce")
        if "year" not in normalized.columns:
            normalized["year"] = normalized["tender_date"].dt.year
    elif "year" in normalized.columns:
        normalized["tender_date"] = pd.to_datetime(normalized["year"].astype(str) + "-01-01")

    price_source = _first_existing(normalized.columns, PRICE_ALIASES)
    if price_source and CANONICAL_PRICE_COLUMN not in normalized.columns:
        normalized[CANONICAL_PRICE_COLUMN] = pd.to_numeric(normalized[price_source], errors="coerce")

    total_source = _first_existing(normalized.columns, TOTAL_ALIASES)
    if total_source and CANONICAL_TOTAL_COLUMN not in normalized.columns:
        normalized[CANONICAL_TOTAL_COLUMN] = pd.to_numeric(normalized[total_source], errors="coerce")

    margin_source = _first_existing(normalized.columns, MARGIN_ALIASES)
    if margin_source and CANONICAL_MARGIN_COLUMN not in normalized.columns:
        normalized[CANONICAL_MARGIN_COLUMN] = pd.to_numeric(normalized[margin_source], errors="coerce")

    cost_source = _first_existing(normalized.columns, COST_ALIASES)
    if cost_source and "estimated_unit_cost" not in normalized.columns:
        normalized["estimated_unit_cost"] = pd.to_numeric(normalized[cost_source], errors="coerce")

    if "buyer_institution_type" not in normalized.columns:
        normalized["buyer_institution_type"] = "Kamu"

    normalized = add_bucket_columns(normalized)
    return normalized


def validate_schema(df: pd.DataFrame) -> SchemaValidationResult:
    normalized = normalize_schema(df)
    required = [*REQUIRED_BASE_COLUMNS, CANONICAL_PRICE_COLUMN, CANONICAL_MARGIN_COLUMN]
    missing = [column for column in required if column not in normalized.columns]
    warnings: list[str] = []

    if not missing:
        if normalized[CANONICAL_PRICE_COLUMN].isna().any():
            warnings.append("Bazı kazanılmış birim fiyat alanları boş veya sayısal değil.")
        if (pd.to_numeric(normalized["quantity"], errors="coerce") <= 0).any():
            warnings.append("Miktar alanında sıfır veya negatif değer var.")
        if normalized["tender_date"].isna().any():
            warnings.append("Bazı ihale tarihleri okunamadı.")

    return SchemaValidationResult(
        valid=not missing,
        missing_columns=missing,
        warnings=warnings,
        row_count=len(normalized),
        columns=list(normalized.columns),
    )


def schema_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    normalized = normalize_schema(df)
    result = validate_schema(normalized)
    numeric_columns = ["quantity", "delivery_months", "competitor_count_estimate", CANONICAL_PRICE_COLUMN]
    null_rates = {
        column: float(normalized[column].isna().mean())
        for column in normalized.columns
        if column in REQUIRED_BASE_COLUMNS or column in numeric_columns
    }
    return {
        "valid": result.valid,
        "missing_columns": result.missing_columns,
        "warnings": result.warnings,
        "row_count": result.row_count,
        "duplicate_tender_ids": int(normalized["tender_id"].duplicated().sum()) if "tender_id" in normalized else 0,
        "null_rates": null_rates,
    }

